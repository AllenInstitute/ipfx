import pandas as pd
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
            sweep_info = self.modify_api_sweep_info(
                sweep_info) if api_sweeps else sweep_info
        else:
            self.notebook = lab_notebook_reader.create_lab_notebook_reader(nwb_file, h5_file)
            sweep_info = self.extract_sweep_meta_data()

        self.sweep_table = pd.DataFrame.from_records(sweep_info)

    def extract_sweep_meta_data(self):
        """

        Returns
        -------
        sweep_props: list of dicts
            where each dict includes sweep properties
        """
        logging.debug("Build sweep table")

        sweep_props = []
        # use same output strategy as h5-nwb converter
        # pick the sampling rate from the first iclamp sweep
        # TODO: figure this out for multipatch
        for sweep_name in self.nwb_data.get_sweep_names():
            sweep_record = {}
            attrs = self.nwb_data.get_sweep_attrs(sweep_name)
            ancestry = attrs["ancestry"]
            sweep_record['clamp_mode'] = ancestry[-1]
            sweep_num = self.nwb_data.get_sweep_number(sweep_name)
            sweep_record['sweep_number'] = sweep_num
            stim_code = self.nwb_data.get_stim_code(sweep_name)
            if not stim_code:
                stim_code = self.notebook.get_value("Stim Wave Name", sweep_num, "")
                logging.debug("Reading stim_code from Labnotebook")
                if len(stim_code) == 0:
                    raise Exception(
                        "Could not read stimulus wave name from lab notebook")

            # stim units are based on timeseries type
            if "CurrentClamp" in ancestry[-1]:
                sweep_record['stimulus_units'] = 'pA'
                sweep_record['clamp_mode'] = 'CurrentClamp'
            elif "VoltageClamp" in ancestry[-1]:
                sweep_record['stimulus_units'] = 'mV'
                sweep_record['clamp_mode'] = 'VoltageClamp'
            else:
                # it's probably OK to skip this sweep and put a 'continue'
                #   here instead of an exception, but wait until there's
                #   an actual error and investigate the data before doing so
                raise Exception(
                    "Unable to determine clamp mode in " + sweep_name)

            # bridge balance
            bridge_balance = self.notebook.get_value(
                "Bridge Bal Value", sweep_num, None)
            sweep_record["bridge_balance_mohm"] = bridge_balance

            # leak_pa (bias current)
            bias_current = self.notebook.get_value(
                "I-Clamp Holding Level", sweep_num, None)
            sweep_record["leak_pa"] = bias_current

            # ephys stim info
            scale_factor = self.notebook.get_value("Scale Factor", sweep_num, None)
            if scale_factor is None:
                raise Exception(
                    "Unable to read scale factor for " + sweep_name)

            sweep_record["stimulus_scale_factor"] = scale_factor

            # PBS-229 change stim name by appending set_sweep_count
            cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
            stim_code_ext = stim_code + "[%d]" % int(cnt)

            sweep_record["stimulus_code_ext"] = stim_code_ext
            sweep_record["stimulus_code"] = stim_code
            sweep_record["stimulus_name"] = self.get_stimulus_name(stim_code)

            sweep_props.append(sweep_record)

        return sweep_props

    def get_sweep_data(self, sweep_number):
        return self.nwb_data.get_sweep_data(sweep_number)
