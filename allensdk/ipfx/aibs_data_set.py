import pandas as pd
import numpy as np

from .ephys_data_set import EphysDataSet, Sweep
from allensdk.core.nwb_data_set import NwbDataSet

class AibsDataSet(EphysDataSet):
    def __init__(self, sweep_list, nwb_file, ontology=None):
        super(AibsDataSet, self).__init__(ontology)
        self.sweep_list = sweep_list
        self.sweep_table = pd.DataFrame.from_records(self.sweep_list)
        self.data_set = NwbDataSet(nwb_file)
        self.nwb_file = nwb_file

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

    
