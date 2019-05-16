import logging

from .ephys_data_set import EphysDataSet

import ipfx.lab_notebook_reader as lab_notebook_reader
import ipfx.nwb_reader as nwb_reader
import ipfx.sweep_props as sp


class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_info=None, nwb_file=None, h5_file=None,
                 ontology=None, api_sweeps=True, validate_stim=True):
        super(AibsDataSet, self).__init__(ontology, validate_stim)

        self.nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_info:
            sweep_info = sp.modify_sweep_info_keys(sweep_info) if api_sweeps else sweep_info
        else:
            self.notebook = lab_notebook_reader.create_lab_notebook_reader(nwb_file, h5_file)
            sweep_info = self.extract_sweep_stim_info()

        self.build_sweep_table(sweep_info)

    def extract_sweep_stim_info(self):
        """

        Returns
        -------
        sweep_info: list of dicts
            where each dict includes sweep properties
        """
        sweep_info = []
        for index, sweep_map in self.nwb_data.sweep_map_table.iterrows():
            sweep_record = {}
            sweep_num = sweep_map["sweep_number"]
            sweep_record['sweep_number'] = sweep_num

            sweep_record['stimulus_units'] = self.get_stimulus_units(sweep_num)

            # bridge balance
            sweep_record["bridge_balance_mohm"] = self.notebook.get_value(
                "Bridge Bal Value", sweep_num, None)

            # leak_pa (bias current)
            sweep_record["leak_pa"] = self.notebook.get_value(
                "I-Clamp Holding Level", sweep_num, None)

            # ephys stim info
            sweep_record["stimulus_scale_factor"] = self.notebook.get_value(
                "Scale Factor", sweep_num, None)

            stim_code = self.get_stimulus_code(sweep_num)
            stim_code_ext = self.get_stimulus_code_ext(stim_code, sweep_num)

            sweep_record["stimulus_code"] = stim_code
            sweep_record["stimulus_code_ext"] = stim_code_ext
            sweep_record["stimulus_name"] = self.get_stimulus_name(stim_code)

            sweep_info.append(sweep_record)

        return sweep_info

    def get_stimulus_code(self, sweep_num):

        stim_code = self.nwb_data.get_stim_code(sweep_num)
        if not stim_code:
            stim_code = self.notebook.get_value("Stim Wave Name", sweep_num, "")
            logging.debug("Reading stim_code from Labnotebook")
            if len(stim_code) == 0:
                raise Exception(
                    "Could not read stimulus wave name from lab notebook")
        return stim_code

    def get_stimulus_code_ext(self, stim_code,sweep_num):
        cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)

        return stim_code_ext

    def get_stimulus_units(self, sweep_num):

        attrs = self.nwb_data.get_sweep_attrs(sweep_num)
        ancestry = attrs["ancestry"]

        # stim units are based on timeseries type
        if "CurrentClamp" in ancestry[-1]:
            units = 'pA'
        elif "VoltageClamp" in ancestry[-1]:
            units = 'mV'
        else:
            raise Exception("Unable to determine clamp mode in {}".format(sweep_num))

        return units

    def get_clamp_mode(self, sweep_num):

        attrs = self.nwb_data.get_sweep_attrs(sweep_num)
        ancestry = attrs["ancestry"]

        if "CurrentClamp" in ancestry[-1]:
            clamp_mode = self.CURRENT_CLAMP
        elif "VoltageClamp" in ancestry[-1]:
            clamp_mode = self.VOLTAGE_CLAMP
        else:
            raise Exception("Unable to determine clamp mode in {}".format(sweep_num))

        return clamp_mode

    def get_sweep_data(self, sweep_number):
        return self.nwb_data.get_sweep_data(sweep_number)

    def get_recording_date(self):
        return self.nwb_data.get_recording_date()
