import re
import warnings
import pandas as pd
from dateutil import parser
import h5py
import numpy as np
from pynwb import NWBHDF5IO
from pynwb.icephys import (CurrentClampSeries, CurrentClampStimulusSeries, VoltageClampSeries,
                           VoltageClampStimulusSeries, IZeroClampSeries)


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning


def get_scalar_value(dataset_from_nwb):
    """
    Some values in NWB are stored as scalar whereas others as np.ndarrays with dimension 1.
    Use this function to retrieve the scalar value itself.
    """

    if isinstance(dataset_from_nwb, np.ndarray):
        return np.asscalar(dataset_from_nwb)

    return dataset_from_nwb


class NwbReader(object):

    def __init__(self, nwb_file):
        self.nwb_file = nwb_file

    def get_sweep_data(self, sweep_number):
        raise NotImplementedError

    def get_sweep_number(self, sweep_name):
        raise NotImplementedError

    def get_stim_code(self, sweep_number):
        raise NotImplementedError

    def get_recording_date(self):
        """
        Extract recording datetime from a session_start_time in nwb
        Use last value if more than one is present

        Returns
        -------
        recording_date: str
            use date format "%Y-%m-%d %H:%M:%S", drop timezone info
        """

        with h5py.File(self.nwb_file, 'r') as f:
            if isinstance(f["session_start_time"].value,np.ndarray): # if ndarray
                session_start_time = f["session_start_time"].value[-1]
            else:
                session_start_time = f["session_start_time"].value # otherwise

            datetime_object = parser.parse(session_start_time)

        recording_date = datetime_object.strftime("%Y-%m-%d %H:%M:%S")

        return recording_date

    @staticmethod
    def get_long_unit_name(unit):
        if unit.startswith('A'):
            return "Amps"
        elif unit.startswith('V'):
            return "Volts"
        else:
            raise ValueError("Unit {} not recognized from TimeSeries".format(unit))

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
                real_sweep_number = read_sweep_from_source(timeseries["source"].value)
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
            return get_scalar_value(sweep_ts["starting_time"].value)

    def get_sweep_attrs(self, sweep_number):

        acquisition_group = self.get_sweep_map(sweep_number)["acquisition_group"]

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_ts = f[self.acquisition_path][acquisition_group]
            attrs = dict(sweep_ts.attrs)

            if self.nwb_major_version == 2:
                for entry in sweep_ts.keys():
                    if entry in ("data", "electrode"):
                        continue

                    attrs[entry] = sweep_ts[entry].value

        return attrs

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
                        if info[i] == 'version':
                            version = info[i+1]
                            break
            toks = version.split('.')
            if len(toks) >= 2:
                major = int(toks[0])
                minor = int(toks[1])
        except:  # noqa: E722
            minor = 0
            major = 0
        return major, minor


class NwbXReader(NwbReader):
    """
    Read data from NWB v2 files created by run_x_to_nwb_conversion.py from
    ABF/DAT files.
    """

    def __init__(self, nwb_file):
        NwbReader.__init__(self, nwb_file)
        self.acquisition_path = "acquisition"
        self.stimulus_path = "stimulus"
        self.nwb_major_version = 2
        self.build_sweep_map()

    def get_sweep_number(self, sweep_name):
        return self.get_real_sweep_number(sweep_name)

    def get_stim_code(self, sweep_number):
        return self.get_sweep_attrs(sweep_number)["stimulus_description"]

    def get_sweep_data(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int
        """

        if not isinstance(sweep_number, (int, np.uint64, np.int64)):
            raise ValueError("sweep_number must be an integer but it is {}".format(type(sweep_number)))

        def getRawDataSourceType(experiment_description):
            """
            Return the original file format of the NWB file
            """

            d = {"abf": False, "dat": False, "unknown": False}

            if experiment_description.startswith("PatchMaster"):
                d["dat"] = True
            elif experiment_description.startswith("Clampex"):
                d["abf"] = True
            else:
                d["unknown"] = True

            return d

        with NWBHDF5IO(self.nwb_file, mode='r') as io:
            nwb = io.read()

            rawDataSourceType = getRawDataSourceType(nwb.experiment_description)
            assert not rawDataSourceType["unknown"], "Can handle data from this raw data source"

            series = nwb.sweep_table.get_series(sweep_number)

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

                    stimulus = s.data[:] * float(s.conversion)
                    stimulus_unit = NwbReader.get_long_unit_name(s.unit)
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


class NwbPipelineReader(NwbReader):
    """
    Reads data from the NWB file generated by the ephys pipeline by converting
    the original NWB generated by MIES.
    """

    def __init__(self, nwb_file):
        NwbReader.__init__(self, nwb_file)
        self.acquisition_path = "acquisition/timeseries"
        self.stimulus_path = "stimulus/presentation"
        self.nwb_major_version = 1
        self.build_sweep_map()

    def get_sweep_data(self, sweep_number):
        """
        Retrieve the stimulus, response, index_range, and sampling rate
        for a particular sweep.  This method hides the NWB file's distinction
        between a "Sweep" and an "Experiment".  An experiment is a subset of
        of a sweep that excludes the initial test pulse.  It also excludes
        any erroneous response data at the end of the sweep (usually for
        ramp sweeps, where recording was terminated mid-stimulus).
        Some sweeps do not have an experiment, so full data arrays are
        returned.  Sweeps that have an experiment return full data arrays
        (include the test pulse) with any erroneous data trimmed from the
        back of the sweep. Data is returned in mV and pA.
        Partially borrowed from the AllenSDK.get_sweep()

        Parameters
        ----------
        sweep_number: int

        Returns
        -------
        dict
            A dictionary with 'stimulus', 'response', 'index_range', and
            'sampling_rate' elements.  The index range is a 2-tuple where
            the first element indicates the end of the test pulse and the
            second index is the end of valid response data.
        """
        with h5py.File(self.nwb_file, 'r') as f:

            sweep_name = 'Sweep_%d' % sweep_number
            swp = f['epochs'][sweep_name]

            sweep_ts = f[self.acquisition_path][sweep_name]

            #   fetch data from file and convert to correct SI unit
            #   this operation depends on file version. early versions of
            #   the file have incorrect conversion information embedded
            #   in the nwb file and data was stored in the appropriate
            #   SI unit. For those files, return uncorrected data.
            #   For newer files (1.1 and later), apply conversion value.

            stimulus_dataset = swp['stimulus/timeseries']['data']
            stimulus_conversion = float(stimulus_dataset.attrs["conversion"])

            response_dataset = swp['response/timeseries']['data']
            response_conversion = float(response_dataset.attrs["conversion"])

            major, minor = self.get_pipeline_version()
            if (major == 1 and minor > 0) or major > 1:
                stimulus = stimulus_dataset.value * stimulus_conversion
                response = response_dataset.value * response_conversion
            else:   # old file version
                stimulus = stimulus_dataset.value
                response = response_dataset.value

            if 'unit' in stimulus_dataset.attrs:
                unit = stimulus_dataset.attrs["unit"].decode('UTF-8')

                unit_str = NwbReader.get_long_unit_name(unit)
            else:
                unit = None
                unit_str = 'Unknown'

            if unit_str == "Amps":
                response *= 1e3,  # voltage, convert units V->mV
                stimulus *= 1e12,  # current, convert units A->pA

            elif unit_str == "Volts":
                response *= 1e12,  # current, convert units A->pA
                stimulus *= 1e3,  # voltage, convert units V->mV

            else:
                raise ValueError("Unknown stimulus unit")

            hz = 1.0 * swp['stimulus/timeseries']['starting_time'].attrs['rate']

            return {
                'stimulus': stimulus,
                'response': response,
                'stimulus_unit': unit_str,
                'sampling_rate': hz
            }

    def get_sweep_number(self, sweep_name):

        sweep_number = self.get_real_sweep_number(sweep_name)
        if sweep_number is None:
            sweep_number = int(sweep_name.split('_')[-1])

        return sweep_number

    def get_stim_code(self, sweep_number):

        acquisition_group = self.get_sweep_map(sweep_number)["acquisition_group"]

        names = ["aibs_stimulus_name", "aibs_stimulus_description"]

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_ts = f[self.acquisition_path][acquisition_group]

            for stimulus_description in names:
                if stimulus_description in sweep_ts.keys():
                    stim_code_raw = sweep_ts[stimulus_description].value
                    stim_code = get_scalar_value(stim_code_raw)

                    if stim_code[-5:] == "_DA_0":
                        return stim_code[:-5]

                    return stim_code


class NwbMiesReader(NwbReader):
    """
    Reads data from the MIES generated NWB file
    """

    def __init__(self, nwb_file):
        NwbReader.__init__(self, nwb_file)
        self.acquisition_path = "acquisition/timeseries"
        self.stimulus_path = "stimulus/presentation"
        self.nwb_major_version = 1
        self.build_sweep_map()

    def get_sweep_data(self, sweep_number):

        sweep_map = self.get_sweep_map(sweep_number)

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_response = f[self.acquisition_path][sweep_map["acquisition_group"]]
            response_dataset = sweep_response["data"]
            hz = 1.0 * sweep_response["starting_time"].attrs['rate']
            sweep_stimulus = f[self.stimulus_path][sweep_map["stimulus_group"]]
            stimulus_dataset = sweep_stimulus["data"]

            response = response_dataset.value
            stimulus = stimulus_dataset.value

            if 'unit' in stimulus_dataset.attrs:
                unit = stimulus_dataset.attrs["unit"].decode('UTF-8')

                unit_str = NwbReader.get_long_unit_name(unit)
            else:
                unit = None
                unit_str = 'Unknown'

        return {"stimulus": stimulus,
                "response": response,
                "sampling_rate": hz,
                "stimulus_unit": unit_str,
                }

    def get_sweep_number(self, sweep_name):

        sweep_number = self.get_real_sweep_number(sweep_name)
        if sweep_number is None:
            sweep_number = int(sweep_name.split('_')[1])

        return sweep_number

    def get_stim_code(self, sweep_number):

        acquisition_group = self.get_sweep_map(sweep_number)["acquisition_group"]

        stimulus_description = "stimulus_description"

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_ts = f[self.acquisition_path][acquisition_group]
            # look for the stimulus description
            if stimulus_description in sweep_ts.keys():
                stim_code_raw = sweep_ts[stimulus_description].value
                stim_code = get_scalar_value(stim_code_raw)

                if stim_code[-5:] == "_DA_0":
                    stim_code = stim_code[:-5]

        return stim_code


def get_nwb_version(nwb_file):
    """
    Return a dict with `major` and `full` NWB version as read from the NWB file.
    """

    with h5py.File(nwb_file, 'r') as f:
        if "nwb_version" in f:         # In v1 this is a dataset
            nwb_version = get_scalar_value(f["nwb_version"].value)
            if nwb_version is not None and re.match("^NWB-1", nwb_version):
                return {"major": 1, "full": nwb_version}

        elif "nwb_version" in f.attrs:   # but in V2 this is an attribute
            nwb_version = f.attrs["nwb_version"]
            if nwb_version is not None and re.match("^2", nwb_version):
                return {"major": 2, "full": nwb_version}

    return {"major": None, "full": None}


def get_nwb1_flavor(nwb_file):
    """
    Determine the flavor of nwb file;
    'Mies':  generated by the MIES hardware with sweeps named as 'data_*'
    'Pipeline': processed by the pipeline to create epoch information and to rename sweeps as 'Sweep_*'
    If sweeps are not present, then assume the original 'Mies' format, since processing had no effect

    Parameters
    ----------
    nwb_file: str
        file name

    Returns
    -------
    str
    """
    with h5py.File(nwb_file, 'r') as f:
        if "acquisition/timeseries" in f:
            sweep_names = [e for e in f["acquisition/timeseries"].keys()]
            if sweep_names:
                sweep_naming_convention = sweep_names[0].split('_')[0]

                if sweep_naming_convention == "Sweep":
                    return "Pipeline"
                elif sweep_naming_convention == "data":
                    return "Mies"
            else:
                return "Mies"
        else:
            sweep_naming_convention = None

    raise ValueError("Unknown sweep naming convention: %s" % sweep_naming_convention)


def create_nwb_reader(nwb_file):
    """Create an appropriate reader of the nwb_file

    Parameters
    ----------
    nwb_file: str file name

    Returns
    -------
    reader object
    """

    nwb_version = get_nwb_version(nwb_file)

    if nwb_version["major"] == 2:
        return NwbXReader(nwb_file)
    elif nwb_version["major"] == 1:
        nwb1_flavor = get_nwb1_flavor(nwb_file)
        if nwb1_flavor == "Mies":
            return NwbMiesReader(nwb_file)
        if nwb1_flavor == "Pipeline":
            return NwbPipelineReader(nwb_file)
    else:
        raise ValueError("Unsupported or unknown NWB major" +
                         "version {} ({})".format(nwb_version["major"], nwb_version["full"]))
