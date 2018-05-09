import pandas as pd
import numpy as np
import re

from .ephys_data_set import EphysDataSet, Sweep
from allensdk.core.nwb_data_set import NwbDataSet


class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_list, nwb_file, ontology=None, api_sweeps=True):
        super(AibsDataSet, self).__init__(ontology)
        self.sweep_list = self.modify_api_sweep_list(sweep_list) if api_sweeps else sweep_list
        self.sweep_table = pd.DataFrame.from_records(self.sweep_list)
        self.data_set = NwbDataSet(nwb_file)
        self.nwb_file = nwb_file

    def modify_api_sweep_list(self, sweep_list):
        return [ { AibsDataSet.SWEEP_NUMBER: s['sweep_number'],
                   AibsDataSet.STIMULUS_UNITS: s['stimulus_units'],
                   AibsDataSet.STIMULUS_AMPLITUDE: s['stimulus_absolute_amplitude'],
                   AibsDataSet.STIMULUS_CODE: re.sub("\[\d+\]", "", s['stimulus_description']),
                   AibsDataSet.STIMULUS_NAME: s['stimulus_name'],
                   AibsDataSet.PASSED: True } for s in sweep_list ]

    def sweep(self, sweep_number):
        data = self.data_set.get_sweep(sweep_number)

        # do it
        hz = data['sampling_rate']
        dt = 1. / hz
        s, e = dt * np.array(data['index_range'])

        return Sweep(t = np.arange(0, len(data['response'])) * dt,
                     v = data['response'] * 1e3, # mV
                     i = data['stimulus'] * 1e12, # pA
                     expt_start = s,
                     expt_end = e,
                     sampling_rate = hz)


