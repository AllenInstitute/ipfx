import h5py
import logging
from allensdk.core.nwb_data_set import NwbDataSet
import stim_features as sf


class NWBReader(object):

    def __init__(self,nwb_file):

        self.nwb_file = nwb_file

    def get_sweep_data(self):
        raise NotImplementedError

    def get_sweep_number(self):
        raise NotImplementedError

    def get_sweep_attrs(self,sweep_name):

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_ts = f["acquisition/timeseries"][sweep_name]
            attrs = dict(sweep_ts.attrs)

        return attrs

    def get_sweep_names(self):

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_names = [e for e in f["acquisition/timeseries"].keys()]

        return sweep_names


class NWB1_0_2Reader(NWBReader):
    """Class for handling NWB version: 1.0.5"""
    def __init__(self, nwb_file):
        NWBReader.__init__(self, nwb_file)


    def get_sweep_data(self, sweep_number):
        """Get sweep data

        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        data: dict of sweep data
        """
        nwb_data = NwbDataSet(self.nwb_file)

        data = nwb_data.get_sweep(sweep_number)

        data['response'] *= 1e3,  # mV
        data['stimulus'] *= 1e12,  # pA

        return data

    def get_sweep_number(self, sweep_name):

        sweep_number = int(sweep_name.split('_')[-1])
        return sweep_number


    def get_stim_code(self, sweep_name):

        stimulus_description_name = "aibs_stimulus_description"

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_ts = f["acquisition/timeseries"][sweep_name]
            # look for the stimulus description
            if stimulus_description_name in sweep_ts.keys():
                stim_code = sweep_ts[stimulus_description_name].value[0]
                if len(stim_code) == 1:
                    stim_code = sweep_ts[stimulus_description_name].value
                else:
                    raise IndexError

        return stim_code


class NWB1_0_5Reader(NWBReader):
    """Class for handling NWB version: 1.0.5"""
    def __init__(self, nwb_file):
        NWBReader.__init__(self, nwb_file)

    def get_sweep_data(self, sweep_number):

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_response = f['acquisition']['timeseries']["data_%05d_AD0" % sweep_number]
            v = sweep_response["data"].value
            hz = 1.0 * sweep_response["starting_time"].attrs['rate']
            sweep_stimulus = f['stimulus']['presentation']["data_%05d_DA0" % sweep_number]
            i = sweep_stimulus["data"].value
            index_range = sf.get_experiment_epoch(i, v, hz)

        return {"stimulus": i,
                "response": v,
                "sampling_rate": hz,
                "index_range": index_range
                }

    def get_sweep_number(self,sweep_name):

        sweep_number = int(sweep_name.split('_')[1])

        return sweep_number



    def get_stim_code(self, sweep_name):

        stimulus_description_name = "stimulus_description"

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_ts = f["acquisition/timeseries"][sweep_name]
            # look for the stimulus description
            if stimulus_description_name in sweep_ts.keys():
                stim_code = sweep_ts[stimulus_description_name].value
                if len(stim_code) == 1:
                    stim_code_raw = sweep_ts[stimulus_description_name].value[0]

                    if stim_code_raw[-5:] == "_DA_0":
                        stim_code = stim_code_raw[:-5]
                    else:
                        stim_code = stim_code_raw
                else:
                    raise IndexError

        return stim_code


def create_nwb_reader(nwb_file):
    """Create an appropriate reader of the nwb_file

    Parameters
    ----------
    nwb_file: str file name

    Returns
    -------
        reader object
    """

    with h5py.File(nwb_file, 'r') as f:
        nwb_version = f['nwb_version'].value

    if nwb_version == "NWB-1.0.2":
        return NWB1_0_2Reader(nwb_file)

    elif nwb_version == "NWB-1.0.5":
        return NWB1_0_5Reader(nwb_file)

    else:
        raise ValueError("Unsupported version of the nwb file")


