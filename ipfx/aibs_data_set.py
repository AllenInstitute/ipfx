import logging

from .ephys_data_set import EphysDataSet

import ipfx.lab_notebook_reader as lab_notebook_reader
import ipfx.nwb_reader as nwb_reader


class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_info=None, nwb_file=None, h5_file=None,
                 ontology=None, api_sweeps=True, validate_stim=True):
        super(AibsDataSet, self).__init__(ontology, validate_stim)

        self.nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_info is not None:
            sweep_info = self.modify_api_sweep_info(sweep_info) if api_sweeps else sweep_info
        else:
            self.notebook = lab_notebook_reader.create_lab_notebook_reader(nwb_file, h5_file)
            sweep_info = self.extract_sweep_meta_data()

        self.build_sweep_table(sweep_info)

    def extract_sweep_meta_data(self):
        """

        Returns
        -------
        sweep_meta_data: list of dicts
            where each dict includes sweep properties
        """
        sweep_meta_data = []

        for sweep_name in self.nwb_data.get_sweep_names():
            sweep_record = {}
            sweep_num = self.nwb_data.get_sweep_number(sweep_name)

            sweep_record['starting_time'] = self.nwb_data.get_starting_time(sweep_name)
            sweep_record['sweep_number'] = sweep_num
            sweep_record['clamp_mode'] = self.get_clamp_mode(sweep_name)
            sweep_record['stimulus_units'] = self.get_stim_units(sweep_name)
            sweep_record["bridge_balance_mohm"] = self.notebook.get_value("Bridge Bal Value", sweep_num, None)
            sweep_record["leak_pa"] = self.notebook.get_value("I-Clamp Holding Level", sweep_num, None)
            sweep_record["stimulus_scale_factor"] = self.get_scale_factor(sweep_num)
            stim_code, stim_code_ext = self.get_stimulus_code(sweep_name)
            sweep_record["stimulus_code"] = stim_code
            sweep_record["stimulus_code_ext"] = stim_code_ext
            sweep_record["stimulus_name"] = self.get_stimulus_name(stim_code)

            sweep_meta_data.append(sweep_record)

        return sweep_meta_data

    def get_stimulus_code(self,sweep_name):

        stim_code = self.nwb_data.get_stim_code(sweep_name)
        sweep_num = self.nwb_data.get_sweep_number(sweep_name)

        if not stim_code:
            stim_code = self.notebook.get_value("Stim Wave Name", sweep_num, "")
            logging.debug("Reading stim_code from Labnotebook")
            if len(stim_code) == 0:
                raise Exception(
                    "Could not read stimulus wave name from lab notebook")

        # PBS-229 change stim name by appending set_sweep_count
        cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)

        return stim_code, stim_code_ext

    def get_scale_factor(self,sweep_num):

        # ephys stim info
        scale_factor = self.notebook.get_value("Scale Factor", sweep_num, None)
        # if scale_factor is None:
        #     raise Exception(
        #         "Unable to read scale factor for " + sweep_name)

        return scale_factor

    def get_stim_units(self, sweep_name):

        attrs = self.nwb_data.get_sweep_attrs(sweep_name)
        ancestry = attrs["ancestry"]

        if "CurrentClamp" in ancestry[-1]:
            return 'pA'
        elif "VoltageClamp" in ancestry[-1]:
            return 'mV'
        else:
            raise Exception("Unknown clamp mode")

    def get_clamp_mode(self, sweep_name):
        attrs = self.nwb_data.get_sweep_attrs(sweep_name)
        ancestry = attrs["ancestry"]

        if "CurrentClamp" in ancestry[-1]:
            return 'CurrentClamp'
        elif "VoltageClamp" in ancestry[-1]:
            return 'VoltageClamp'
        else:
            # it's probably OK to skip this sweep and put a 'continue'
            #   here instead of an exception, but wait until there's
            #   an actual error and investigate the data before doing so
            raise Exception(
                "Unable to determine clamp mode in " + sweep_name)

    def get_sweep_data(self, sweep_number):
        return self.nwb_data.get_sweep_data(sweep_number)
