import pandas as pd
import numpy as np
import re
import logging

from .ephys_data_set import EphysDataSet, Sweep
import ipfx.lab_notebook_reader as lab_notebook_reader
import ipfx.nwb_reader as nwb_reader


class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_info=None, nwb_file=None, h5_file=None, ontology=None, api_sweeps=True):
        super(AibsDataSet, self).__init__(ontology)
        self.nwb_file = nwb_file
        self.h5_file = h5_file
        self.nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_info is not None:
            sweep_info = self.modify_api_sweep_info(
                sweep_info) if api_sweeps else sweep_info
        else:
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

        notebook = lab_notebook_reader.create_lab_notebook_reader(
            self.nwb_file, self.h5_file)

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
                stim_code = notebook.get_value("Stim Wave Name", sweep_num, "")
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
            bridge_balance = notebook.get_value(
                "Bridge Bal Value", sweep_num, None)
            sweep_record["bridge_balance_mohm"] = bridge_balance

            # leak_pa (bias current)
            bias_current = notebook.get_value(
                "I-Clamp Holding Level", sweep_num, None)
            sweep_record["leak_pa"] = bias_current

            # ephys stim info
            scale_factor = notebook.get_value("Scale Factor", sweep_num, None)
            if scale_factor is None:
                raise Exception(
                    "Unable to read scale factor for " + sweep_name)

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


            sweep_props.append(sweep_record)

        return sweep_props

    def sweep(self, sweep_number):
        """
        Create an instance of the Sweep object from a data set sweep
        Time t=0 is set to the start of the experiment epoch

        Parameters
        ----------
        sweep_number

        Returns
        -------
        Sweep object
        """

        sweep_data = self.nwb_data.get_sweep_data(sweep_number)
        hz = sweep_data['sampling_rate']
        dt = 1. / hz
        sweep_info = self.get_sweep_info_by_sweep_number(sweep_number)

        start_ix, end_ix = sweep_data['index_range']

        t = np.arange(0, end_ix+1)*dt - start_ix*dt

        response = sweep_data['response'][0:end_ix+1]
        stimulus = sweep_data['stimulus'][0:end_ix+1]

        clamp_mode = sweep_info.get('clamp_mode', None)
        if clamp_mode is None:
            clamp_mode = "CurrentClamp" if sweep_info[
                'stimulus_units'] in self.ontology.current_clamp_units else "VoltageClamp"

        if clamp_mode == "VoltageClamp":  # voltage clamp
            v = stimulus
            i = response
        elif clamp_mode == "CurrentClamp":  # Current clamp
            v = response
            i = stimulus
        else:
            raise ValueError("Incorrect stimulus unit")

        try:
            sweep = Sweep(t=t,
                          v=v,
                          i=i,
                          sampling_rate=sweep_data['sampling_rate'],
                          expt_idx_range=sweep_data['index_range'],
                          sweep_number=sweep_number,
                          clamp_mode=clamp_mode
                          )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep
