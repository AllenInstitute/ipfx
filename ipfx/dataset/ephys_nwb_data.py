from typing import Dict, Any, List, Optional
import warnings
import re

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

from io import BytesIO
import h5py
from pynwb import NWBHDF5IO

from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries)
from ipfx.py2to3 import to_str

from ipfx.dataset.stimulus import StimulusOntology
from ipfx.dataset.ephys_data_interface import EphysDataInterface


def get_scalar_value(dataset_from_nwb):
    """
    Some values in NWB are stored as scalar whereas others as np.ndarrays with dimension 1.
    Use this function to retrieve the scalar value itself.
    """

    if isinstance(dataset_from_nwb, np.ndarray):
        return dataset_from_nwb.item()

    return dataset_from_nwb


def get_finite_or_none(d, key):
    try:
        value = d[key]
    except KeyError:
        return None

    if np.isnan(value):
        return None

    return value


class EphysNWBData(EphysDataInterface):
    """
    Abstract base class for implementing an EphysDataInterface with an NWB file

    Provides common NWB2 reading and writing functionality

    """

    SCALAR_ATTRS = ["bias_current",
                    "stimulus_scale_factor",
                    "bridge_balance",
                    "stimulus_scale_factor"]

    def __init__(self,
                 nwb_file: str,
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 validate_stim: bool = True,
                 ):

        super().__init__(ontology=ontology)
        self.nwb_file = nwb_file
        
        if load_into_memory:
            with open(nwb_file, 'rb') as fh:
                data = BytesIO(fh.read())

            _h5_file = h5py.File(data, "r")
            self.nwb = NWBHDF5IO(path=_h5_file.filename, mode="r+",file=_h5_file).read()
        else:
            self.nwb = NWBHDF5IO(nwb_file, mode='r').read()

        self.acquisition_path = "acquisition"
        self.stimulus_path = "stimulus/presentation"
        self.nwb_major_version = 2
        self.build_sweep_map()

    def get_sweep_data(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int
        """

        if not isinstance(sweep_number, (int, np.uint64, np.int64)):
            raise ValueError("sweep_number must be an integer but it is {}".format(type(sweep_number)))

        series = self.nwb.sweep_table.get_series(sweep_number)

        if series is None:
            raise ValueError("No TimeSeries found for sweep number {}.".format(sweep_number))

        # we need one "*ClampStimulusSeries" and one "*ClampSeries"

        response = None
        stimulus = None
        for s in series:

            if isinstance(s, (VoltageClampSeries, CurrentClampSeries, IZeroClampSeries)):
                if response is not None:
                    raise ValueError("Found multiple response TimeSeries in NWB file for sweep number {}.".format(sweep_number))

                response = s.data[:] * float(s.conversion)
            elif isinstance(s, (VoltageClampStimulusSeries, CurrentClampStimulusSeries)):
                if stimulus is not None:
                    raise ValueError("Found multiple stimulus TimeSeries in NWB file for sweep number {}.".format(sweep_number))

                    response = s.data[:] * float(s.conversion)
                    response_unit = self.get_long_unit_name(s.unit)
                    self.validate_SI_unit(response_unit)

                elif isinstance(s, (VoltageClampStimulusSeries, CurrentClampStimulusSeries)):
                    if stimulus is not None:
                        raise ValueError("Found multiple stimulus TimeSeries in NWB file for sweep number {}.".format(sweep_number))

                    stimulus = s.data[:] * float(s.conversion)
                    stimulus_unit = self.get_long_unit_name(s.unit)
                    self.validate_SI_unit(stimulus_unit)

                    stimulus_rate = float(s.rate)
                else:
                    raise ValueError("Unexpected TimeSeries {}.".format(type(s)))

        if stimulus is None:
            raise ValueError("Could not find one stimulus TimeSeries for sweep number {}.".format(sweep_number))
        elif response is None:
            raise ValueError("Could not find one response TimeSeries for sweep number {}.".format(sweep_number))

        return {
            'stimulus': stimulus,
            'response': response,
            'stimulus_unit': stimulus_unit,
            'sampling_rate': stimulus_rate
        }

    def get_sweep_attrs(self, sweep_number):

        acquisition_group = self.get_sweep_map(sweep_number)["acquisition_group"]

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_ts = f[self.acquisition_path][acquisition_group]
            attrs = dict(sweep_ts.attrs)

            if self.nwb_major_version == 2:
                for entry in sweep_ts.keys():
                    if entry in ("data", "electrode"):
                        continue

                    attrs[entry] = sweep_ts[entry][()]

        for k in self.SCALAR_ATTRS:
            if k in attrs.keys():
                attrs[k] = get_scalar_value(attrs[k])

        return attrs

    def get_sweep_number(self, sweep_name):
        return self.get_real_sweep_number(sweep_name)

    def get_stim_code(self, sweep_number):
        stim_code = self.get_sweep_attrs(sweep_number)["stimulus_description"]
        if stim_code[-5:] == "_DA_0":
            stim_code = stim_code[:-5]
        return stim_code.split("[")[0]

    def get_spike_times(self, sweep_number):

        spikes = self.nwb.get_processing_module('spikes')
        sweep_spikes = spikes.get_data_interface(f"Sweep_{sweep_number}")

        return sweep_spikes.timestamps

    def get_session_start_time(self):
        """
        Extract session_start_time in nwb
        Use last value if more than one is present

        Returns
        -------
        recording_date: str
            use date format "%Y-%m-%d %H:%M:%S", drop timezone info
        """

        with h5py.File(self.nwb_file, 'r') as f:
            if isinstance(f["session_start_time"][()],np.ndarray): # if ndarray
                session_start_time = f["session_start_time"][()][-1]
            else:
                session_start_time = f["session_start_time"][()] # otherwise

            datetime_object = dateparser.parse(session_start_time)

        return datetime_object

    def get_recording_date(self):

        datetime_object = self.get_session_start_time()

        recording_date = datetime_object.strftime("%Y-%m-%d %H:%M:%S")

        return recording_date

    def get_stimulus_unit(self, sweep_number):

        sweep_map = self.get_sweep_map(sweep_number)

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_stimulus = f[self.stimulus_path][sweep_map["stimulus_group"]]
            stimulus_dataset = sweep_stimulus["data"]

            unit = self.get_unit_name(stimulus_dataset.attrs)
            unit_str = self.get_long_unit_name(unit)

            return unit_str

    @staticmethod
    def get_unit_name(stim_attrs):
        if 'unit' in stim_attrs:
            unit = to_str(stim_attrs["unit"])
        elif 'units' in stim_attrs:
            unit = to_str(stim_attrs["units"])
        else:
            unit = None

        return unit

    @staticmethod
    def get_long_unit_name(unit):
        if not unit:
            return "Unknown"
        elif unit in ["Amps", "A", "amps"]:
            return "Amps"
        elif unit in ["Volts", "V", "volts"]:
            return "Volts"
        else:
            return unit

    @staticmethod
    def validate_SI_unit(unit):

        valid_SI_units = ["Volts", "Amps", "amperes"]
        if unit not in valid_SI_units:
            raise ValueError(F"Unit {unit} is not among the valid SI units {valid_SI_units}")

    def get_real_sweep_number(self, sweep_name, assumed_sweep_number=None):
        """
        Return the real sweep number for the given sweep_name. Falls back to
        assumed_sweep_number if given.
        """

        with h5py.File(self.nwb_file, 'r') as f:
            timeseries = f[self.acquisition_path][sweep_name]

            real_sweep_number = None

            def read_sweep_from_source(source):
                source = get_scalar_value(source)
                for x in source.split(";"):
                    result = re.search(r"^Sweep=(\d+)$", x)
                    if result:
                        return int(result.group(1))

            if "source" in timeseries:
                real_sweep_number = read_sweep_from_source(timeseries["source"][()])
            elif "source" in timeseries.attrs:
                real_sweep_number = read_sweep_from_source(timeseries.attrs["source"])
            elif "sweep_number" in timeseries.attrs:
                real_sweep_number = timeseries.attrs["sweep_number"]

            if real_sweep_number is None:
                warnings.warn("Sweep number not found, returning: None")

            return real_sweep_number

    def get_starting_time(self, data_set_name):
        with h5py.File(self.nwb_file, 'r') as f:
            sweep_ts = f[self.acquisition_path][data_set_name]
            return get_scalar_value(sweep_ts["starting_time"][()])

    def build_sweep_map(self):
        """
        Build table for mapping sweep_number to the names of stimulus and acquisition groups in the nwb file
        Returns
        -------
        """

        sweep_map = []

        for stim_group, acq_group in zip(self.get_stimulus_groups(), self.get_acquisition_groups()):
            sweep_record = {}
            sweep_record["acquisition_group"] = acq_group
            sweep_record["stimulus_group"] = stim_group
            sweep_record["sweep_number"] = self.get_sweep_number(acq_group)
            sweep_record["starting_time"] = self.get_starting_time(acq_group)

            sweep_map.append(sweep_record)

        self.sweep_map_table = pd.DataFrame.from_records(sweep_map)

        if sweep_map:
            self.drop_reacquired_sweeps()

    def drop_reacquired_sweeps(self):
        """
        If sweep was re-acquired, then drop earlier acquired sweep with the same sweep_number
        """
        self.sweep_map_table.sort_values(by="starting_time")
        duplicates = self.sweep_map_table.duplicated(subset="sweep_number",keep="last")
        reacquired_sweep_numbers = self.sweep_map_table[duplicates]["sweep_number"].values

        if len(reacquired_sweep_numbers) > 0:
            warnings.warn("Sweeps {} were reacquired. Keeping acquisitions of sweeps with the latest staring time.".
                          format(reacquired_sweep_numbers))

        self.sweep_map_table.drop_duplicates(subset="sweep_number", keep="last",inplace=True)

    def get_sweep_names(self):

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_names = [e for e in f[self.acquisition_path].keys()]

        return sweep_names

    def get_sweep_map(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int
            real sweep number
        Returns
        -------
        sweep_map: dict
        """
        if sweep_number is not None:
            mask = self.sweep_map_table["sweep_number"] == sweep_number
            st = self.sweep_map_table[mask]
            return st.to_dict(orient='records')[0]
        else:
            raise ValueError("Invalid sweep number {}".format(sweep_number))

    def get_acquisition_groups(self):

        with h5py.File(self.nwb_file, 'r') as f:
            if self.acquisition_path in f:
                acquisition_groups = [e for e in f[self.acquisition_path].keys()]
            else:
                acquisition_groups = []

        return acquisition_groups

    def get_stimulus_groups(self):

        with h5py.File(self.nwb_file, 'r') as f:
            if self.stimulus_path in f:
                stimulus_groups = [e for e in f[self.stimulus_path].keys()]
            else:
                stimulus_groups = []
        return stimulus_groups

    def get_pipeline_version(self):
        """ Returns the AI pipeline version number, stored in the
            metadata field 'generated_by'. If that field is
            missing, version 0.0 is returned.
            Borrowed from the AllenSDK

            Returns
            -------
            int tuple: (major, minor)
        """
        try:
            with h5py.File(self.nwb_file, 'r') as f:
                if 'generated_by' in f["general"]:
                    info = f["general/generated_by"]
                    # generated_by stores array of keys and values
                    # keys are even numbered, corresponding values are in
                    #   odd indices
                    for i in range(len(info)):
                        if to_str(info[i]) == 'version':
                            version = to_str(info[i+1])
                            break
            toks = version.split('.')
            if len(toks) >= 2:
                major = int(toks[0])
                minor = int(toks[1])
        except:  # noqa: E722
            minor = 0
            major = 0
        return major, minor
