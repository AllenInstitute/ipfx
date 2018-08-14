import h5py
import sys
import logging
import numpy as np
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


class NWB_Processed(NWBReader):
    def __init__(self, nwb_file):
        NWBReader.__init__(self, nwb_file)


    def get_sweep_data(self, sweep_number):
        """Get sweep data and convert current to pA and voltage to mV

        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        data: dict of sweep data in response (mV) and stimulus (pA)
        """
        nwb_data = NwbDataSet(self.nwb_file)

        data = nwb_data.get_sweep(sweep_number)

        if data['stimulus_unit']=="Amps":
            data['response'] *= 1e3,  # voltage, covert units V->mV
            data['stimulus'] *= 1e12,  # current, convert units A->pA

        elif data['stimulus_unit']=="Volts":
            data['response'] *= 1e12,  # current, convert units A->pA
            data['stimulus'] *= 1e3,  # voltage, covert units V->mV

        else:
            raise ValueError("Incorrect stimulus unit")

        return data

    def get_sweep_number(self, sweep_name):

        sweep_number = int(sweep_name.split('_')[-1])
        return sweep_number

    def get_stim_code(self, sweep_name):

        stimulus_description = "aibs_stimulus_description"

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_ts = f["acquisition/timeseries"][sweep_name]
            # look for the stimulus description
            if stimulus_description in sweep_ts.keys():
                stim_code_raw = sweep_ts[stimulus_description].value

                if type(stim_code_raw) is np.ndarray:
                    stim_code = str(stim_code_raw[0])
                else:
                    stim_code = str(stim_code_raw)

                if stim_code[-5:] == "_DA_0":
                    stim_code = stim_code[:-5]

        return stim_code


class NWB_Raw(NWBReader):
    def __init__(self, nwb_file):
        NWBReader.__init__(self, nwb_file)

    def get_sweep_data(self, sweep_number):

        with h5py.File(self.nwb_file, 'r') as f:
            sweep_response = f['acquisition']['timeseries']["data_%05d_AD0" % sweep_number]
            response = sweep_response["data"].value
            hz = 1.0 * sweep_response["starting_time"].attrs['rate']
            sweep_stimulus = f['stimulus']['presentation']["data_%05d_DA0" % sweep_number]
            stimulus = sweep_stimulus["data"].value

            if 'unit' in sweep_stimulus["data"].attrs:
                unit = sweep_stimulus["data"].attrs["unit"].decode('UTF-8')

                unit_str = None
                if unit.startswith('A'):
                    unit_str = "Amps"
                elif unit.startswith('V'):
                    unit_str = "Volts"
                assert unit_str is not None, Exception("Stimulus time series unit not recognized")
            else:
                unit_str = 'Unknown'

        return {"stimulus": stimulus,
                "response": response,
                "sampling_rate": hz,
                "stimulus_unit": unit_str
                }

    def get_sweep_number(self,sweep_name):

        sweep_number = int(sweep_name.split('_')[1])

        return sweep_number



    def get_stim_code(self, sweep_name):

        stimulus_description = "stimulus_description"

        with h5py.File(self.nwb_file, 'r') as f:

            sweep_ts = f["acquisition/timeseries"][sweep_name]
            # look for the stimulus description
            if stimulus_description in sweep_ts.keys():
                stim_code_raw = sweep_ts[stimulus_description].value

                if type(stim_code_raw) is np.ndarray:
                    stim_code = str(stim_code_raw[0])
                else:
                    stim_code = str(stim_code_raw)

                if stim_code[-5:] == "_DA_0":
                    stim_code = stim_code[:-5]


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
        epochs = f["epochs"].keys()

    if epochs:
        return NWB_Processed(nwb_file)
    else:
        return NWB_Raw(nwb_file)



