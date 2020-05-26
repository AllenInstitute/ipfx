from typing import Dict, Any, Tuple, Optional, Sequence, Callable
import warnings

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

from io import BytesIO
import h5py
from pynwb import NWBHDF5IO

from pynwb.icephys import (
    CurrentClampSeries, CurrentClampStimulusSeries,
    VoltageClampSeries, VoltageClampStimulusSeries,
    IZeroClampSeries, PatchClampSeries)

from ipfx.stimulus import StimulusOntology
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

    if value is None or np.isnan(value):
        return None

    return value


class EphysNWBData(EphysDataInterface):
    """
    Abstract base class for implementing an EphysDataInterface with an NWB file

    Provides common NWB2 reading and writing functionality

    """

    STIMULUS = (VoltageClampStimulusSeries, CurrentClampStimulusSeries)
    RESPONSE = (VoltageClampSeries, CurrentClampSeries)

    def __init__(self,
                 nwb_file: None,
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 validate_stim: bool = True
                 ):

        super(EphysNWBData, self).__init__(
            ontology=ontology, validate_stim=validate_stim)
        self.load_nwb(nwb_file, load_into_memory)
        
        self.acquisition_path = "acquisition"
        self.stimulus_path = "stimulus/presentation"
        self.nwb_major_version = 2
        
    def load_nwb(self, nwb_file: None, load_into_memory: bool = True):
        """
        Load NWB to self.nwb
        
        Parameters
        ----------
        nwb_file: NWB file path or hdf5 obj
        load_into_memory: whether using load_into_memory approach to load NWB
        """

        self.nwb_file = nwb_file
        if isinstance(nwb_file, str):
            if load_into_memory:
                with open(nwb_file, 'rb') as fh:
                    data = BytesIO(fh.read())
                _h5_file = h5py.File(data, "r")
                reader = NWBHDF5IO(path=_h5_file.filename, mode="r",file=_h5_file, load_namespaces=True)
            else:
                reader = NWBHDF5IO(nwb_file, mode='r', load_namespaces=True)
        elif isinstance(nwb_file, BytesIO):
            _h5_file = h5py.File(nwb_file, "r")
            reader = NWBHDF5IO(path=_h5_file.filename, mode="r",file=_h5_file, load_namespaces=True)
        else:
            raise TypeError("Invalid input NWB file (only accept NWB filepath or hdf5 obj)!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.nwb = reader.read()


    def _get_series(self, sweep_number: int,
                    series_class: Tuple[PatchClampSeries]):
        """
        Get Time Series of a specified class
        Parameters
        ----------
        sweep_number: int sweep number
        series_class: pynwb.PatchClampSeries

        Returns
        -------
        series: pynwb.PatchClampSeries
        """
        series = self.nwb.sweep_table.get_series(sweep_number)

        if series is None:
            raise ValueError("No TimeSeries found for sweep number {}.".format(sweep_number))

        matching_series = []

        for s in series:
            if isinstance(s,series_class):
                matching_series.append(s)

        if len(matching_series) == 1:
            return matching_series[0]
        else:
            raise ValueError("Found multiple stimulus series "
                             "{[s.name for s in matching_series]} "
                             "for sweep number {sweep_number}")

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

        if stimulus_unit == "Volts":
            stimulus = stimulus * 1.0e3
            response = response * 1.0e12 
        elif stimulus_unit == "Amps":
            stimulus = stimulus * 1.0e12
            response = response * 1.0e3

        return {
            'stimulus': stimulus,
            'response': response,
            'stimulus_unit': stimulus_unit,
            'sampling_rate': stimulus_rate
        }

    def get_sweep_attrs(self, sweep_number):

        rs = self._get_series(sweep_number, self.RESPONSE)

        if isinstance(rs, VoltageClampSeries):
            attrs = {
                'gain': rs.gain,
                'stimulus_description': rs.stimulus_description,
                'sweep_number': sweep_number,
                'clamp_mode': "VoltageClamp"
            }

        elif isinstance(rs,CurrentClampSeries):

            attrs = {
                'gain': rs.gain,
                'stimulus_description': rs.stimulus_description,
                'sweep_number': sweep_number,
                'bias_current': rs.bias_current,
                'bridge_balance': rs.bridge_balance,
                'clamp_mode': "CurrentClamp"
            }
        else:
            raise ValueError(f"Must be response series {self.RESPONSE} ")
        return attrs

    def get_sweep_metadata(self, sweep_number: int):
        return NotImplementedError

    @property
    def sweep_numbers(self) -> Sequence[int]:
        return np.unique(self.nwb.sweep_table.sweep_number[:])

    def get_stimulus_code(self, sweep_number):
        rs = self._get_series(sweep_number, self.RESPONSE)
        stim_code = rs.stimulus_description
        if stim_code[-5:] == "_DA_0":
            stim_code = stim_code[:-5]
        return stim_code.split("[")[0]

    def get_full_recording_date(self):
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


    def get_sweep_metadata(self, sweep_number: int):
        raise NotImplementedError

    def get_stimulus_unit(self,sweep_number):
        stimulus_series = self._get_series(sweep_number,self.STIMULUS)
        return type(self).get_long_unit_name(stimulus_series.unit)

    def get_clamp_mode(self,sweep_number):
        return self.get_sweep_attrs(sweep_number)["clamp_mode"]

    def get_spike_times(self, sweep_number):
        spikes = self.nwb.get_processing_module('spikes')
        sweep_spikes = spikes.get_data_interface(f"Sweep_{sweep_number}")
        return sweep_spikes.timestamps

    @staticmethod
    def get_long_unit_name(unit):
        if not unit:
            return "Unknown"
        elif unit in ["Amps", "A", "amps", "amperes"]:
            return "Amps"
        elif unit in ["Volts", "V", "volts"]:
            return "Volts"
        else:
            return unit

    @staticmethod
    def validate_SI_unit(unit):

        valid_SI_units = ["Volts", "Amps"]
        if unit not in valid_SI_units:
            raise ValueError(F"Unit {unit} is not among the valid SI units {valid_SI_units}")
