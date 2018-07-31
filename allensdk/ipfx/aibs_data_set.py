import pandas as pd
import numpy as np
import re
import logging

from .ephys_data_set import EphysDataSet, Sweep
import allensdk.ipfx.mies_nwb.lab_notebook_reader as lab_notebook_reader
import allensdk.ipfx.nwb_reader as nwb_reader

import allensdk.ipfx.stim_features as st

class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_props=[], nwb_file=None, h5_file=None, ontology=None, api_sweeps=True):
        super(AibsDataSet, self).__init__(ontology)
        self.nwb_file = nwb_file
        self.h5_file = h5_file
        self.nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_props:
            sweep_props = self.modify_api_sweep_props(sweep_props) if api_sweeps else sweep_props
            self.sweep_table = pd.DataFrame.from_records(sweep_props)
        else:
            sweep_props = self.extract_sweep_props()
            self.sweep_table = pd.DataFrame.from_records(sweep_props)
            self.sweep_table.to_csv("sweep_table_with_completed.csv", sep=" ", index=False,na_rep="NA")

    def extract_sweep_props(self):
        """
        :parameter:

        :return:
            dict of sweep properties
        """
        notebook = lab_notebook_reader.create_lab_notebook_reader(self.nwb_file, self.h5_file)

        sweep_props = []
        logging.debug("*************Building sweep properties tables***********************")

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
                stim_code = notebook.get_value("Stim Wave Name", sweep_num, "")
                logging.debug("Reading stim_code from Labnotebook")
                if len(stim_code) == 0:
                    raise Exception("Could not read stimulus wave name from lab notebook")

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

            # if (sweep_record["clamp_mode"] =='CurrentClamp') and (sweep_record["stimulus_name"] not in (self.search_names+self.test_names)):
            #
            #         sweep_data = self.nwb_data.get_sweep_data(sweep_record["sweep_number"])
            #
            #         i = sweep_data["stimulus"]
            #         v = sweep_data["response"]
            #         hz = sweep_data["sampling_rate"]
            #
            #         sweep_record["truncated"] = not(st.sweep_completion_check(i, v, hz))
            # else:
            #     sweep_record["truncated"] = None

            sweep_props.append(sweep_record)

        return sweep_props

    def modify_api_sweep_list(self, sweep_list):

        return [ { AibsDataSet.SWEEP_NUMBER: s['sweep_number'],
                   AibsDataSet.STIMULUS_UNITS: s['stimulus_units'],
                   AibsDataSet.STIMULUS_AMPLITUDE: s['stimulus_absolute_amplitude'],
                   AibsDataSet.STIMULUS_CODE: re.sub("\[\d+\]", "", s['stimulus_description']),
                   AibsDataSet.STIMULUS_NAME: s['stimulus_name'],
                   AibsDataSet.PASSED: True } for s in sweep_list ]

    def sweep(self, sweep_number, full_sweep = False):
        """

        Parameters
        ----------
        sweep_number
        full_sweep

        Returns
        -------

        """

        sweep_data = self.nwb_data.get_sweep_data(sweep_number)
        hz = sweep_data['sampling_rate']
        dt = 1. / hz
        sweep_data['time'] = np.arange(0, len(sweep_data['response'])) * dt
        assert len(sweep_data['response']) == len(sweep_data['stimulus']), "Stimulus and response have different duration"

        if full_sweep:
            end_ix = len(sweep_data['response'])
        else:
            end_ix = sweep_data['index_range'][1]   # cut off at the end of the experiment epoch

        try:
            sweep = Sweep(t = sweep_data['time'][0:end_ix],
                          v = sweep_data['response'][0:end_ix], # mV
                          i = sweep_data['stimulus'][0:end_ix], # pA
                          sampling_rate = sweep_data['sampling_rate'],
                          expt_idx_range = sweep_data['index_range'],
                          id = sweep_number,
                          )

        except Exception as e:
            logging.warning("Error reading sweep %d" % sweep_num)
            raise

        return sweep
