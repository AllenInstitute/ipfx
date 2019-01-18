import numpy as np


class Sweep(object):
    def __init__(self, t, v, i, expt_idx_range, sampling_rate=None, sweep_number=None):
        self.t = t
        self.v = v
        self.i = i
        self.expt_idx_range = expt_idx_range
        self.sampling_rate = sampling_rate
        self.sweep_number = sweep_number

    @property
    def t_end(self):
        return self.t[-1]

    @property
    def expt_t_range(self):
        dt = 1. / self.sampling_rate
        return dt * np.array(self.expt_idx_range)


class SweepSet(object):
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def _prop(self, prop):
        return [getattr(s, prop) for s in self.sweeps]

    @property
    def t(self):
        return self._prop('t')

    @property
    def v(self):
        return self._prop('v')

    @property
    def i(self):
        return self._prop('i')

    @property
    def expt_start(self):
        return self._prop('expt_start')

    @property
    def expt_end(self):
        return self._prop('expt_end')

    @property
    def expt_t_range(self):
        return self._prop('expt_t_range')

    @property
    def t_end(self):
        return self._prop('t_end')

    @property
    def sweep_number(self):
        return self._prop('sweep_number')
