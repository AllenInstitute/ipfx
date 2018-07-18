import pandas as pd
import numpy as np
import re
import h5py
import logging

from .ephys_data_set import EphysDataSet, Sweep
from allensdk.core.nwb_data_set import NwbDataSet
import allensdk.ipfx.mies_nwb.lab_notebook_reader as lab_notebook_reader
import allensdk.ipfx.nwb_data_reader as nwb_data_reader


class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_list=[], nwb_file=None, h5_file=None, ontology=None, api_sweeps=True):
        super(AibsDataSet, self).__init__(ontology)
        self.data_set = NwbDataSet(nwb_file)
        self.nwb_file = nwb_file
        self.h5_file = h5_file
        self.nwb_data = nwb_data_reader.create_nwb_data_reader(nwb_file)

        if sweep_list:

            self.sweep_list = self.modify_api_sweep_list(sweep_list) if api_sweeps else sweep_list
            self.sweep_table = pd.DataFrame.from_records(self.sweep_list)
#            self.sweep_table.to_csv("sweep_table_pass.csv", sep=" ", index=False) # for debugging

        else:
            self.sweep_table = self.build_sweep_table()
#            self.sweep_table.to_csv("sweep_table.csv", sep=" ", index=False) # for debugging

    def build_sweep_table(self):
        """
        :parameter:

        :return:
            Data Frame of sweeps data
        """
        notebook = lab_notebook_reader.create_lab_notebook_reader(self.nwb_file, self.h5_file)

        nwbf = h5py.File(self.nwb_file, 'r')

        sweep_data = []

        # use same output strategy as h5-nwb converter
        # pick the sampling rate from the first iclamp sweep
        # TODO: figure this out for multipatch
        for sweep_name in nwbf["acquisition/timeseries"]:
            sweep_record = {}
            sweep_ts = nwbf["acquisition/timeseries"][sweep_name]

            ancestry = sweep_ts.attrs["ancestry"]
            sweep_record['clamp_mode'] = ancestry[-1]
            #            sweep_num = self.get_sweep_number(sweep_name)
            sweep_num = self.nwb_data.get_sweep_number(sweep_name)
            sweep_record['sweep_number'] = sweep_num

            stim_code = self.nwb_data.get_stim_code(sweep_name)
            if not stim_code:
                stim_code = notebook.get_value("Stim Wave Name", sweep_num, "")
                logging.debug("Reading stim_code from Labnotebook")
                if len(stim_code) == 0:
                    raise Exception("Could not read stimulus wave name from lab notebook")

            # stim units are based on timeseries type
            ancestry = sweep_ts.attrs["ancestry"]
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
                raise Exception("Unable to determine clamp mode in " + sweep_name)

            # bridge balance
            bridge_balance = notebook.get_value("Bridge Bal Value", sweep_num, None)
            sweep_record["bridge_balance_mohm"] = bridge_balance

            # leak_pa (bias current)
            bias_current = notebook.get_value("I-Clamp Holding Level", sweep_num, None)
            sweep_record["leak_pa"] = bias_current

            # ephys stim info
            scale_factor = notebook.get_value("Scale Factor", sweep_num, None)
            if scale_factor is None:
                raise Exception("Unable to read scale factor for " + sweep_name)

            sweep_record["stimulus_scale_factor"] = scale_factor

            # PBS-229 change stim name by appending set_sweep_count
            cnt = notebook.get_value("Set Sweep Count", sweep_num, 0)
            stim_code_ext = stim_code + "[%d]" % int(cnt)

            sweep_record["stimulus_code_ext"] = stim_code_ext
            sweep_record["stimulus_code"] = stim_code

            if self.ontology:
                # make sure we can find all of our stimuli in the ontology
                stim = self.ontology.find_one(stim_code, tag_type='code')
                sweep_record["stimulus_name"] = stim.tags(tag_type='name')[0][-1]

            sweep_data.append(sweep_record)

        nwbf.close()
        return pd.DataFrame.from_records(sweep_data)

    def modify_api_sweep_list(self, sweep_list):

        return [ { AibsDataSet.SWEEP_NUMBER: s['sweep_number'],
                   AibsDataSet.STIMULUS_UNITS: s['stimulus_units'],
                   AibsDataSet.STIMULUS_AMPLITUDE: s['stimulus_absolute_amplitude'],
                   AibsDataSet.STIMULUS_CODE: re.sub("\[\d+\]", "", s['stimulus_description']),
                   AibsDataSet.STIMULUS_NAME: s['stimulus_name'],
                   AibsDataSet.PASSED: True } for s in sweep_list ]

    def sweep(self, sweep_number):

        sweep_data = self.nwb_data.get_sweep_data(sweep_number)
        hz = sweep_data['sampling_rate']
        dt = 1. / hz
        sweep_data['time'] = np.arange(0, len(sweep_data['response'])) * dt
        assert len(sweep_data['response']) == len(sweep_data['stimulus']), "Stimulus and response have different duration"

        return Sweep(t = sweep_data['time'],
                     v = sweep_data['response'], # mV
                     i = sweep_data['stimulus'], # pA
                     sampling_rate = sweep_data['sampling_rate'],
                     expt_idx_range = sweep_data['index_range'],
                     id = sweep_number,
                     )
