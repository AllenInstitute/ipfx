import numpy as np


class Sweep(object):
    def __init__(self, t, v, i, sampling_rate=None, sweep_number=None):
        self.t = t
        self.v = v
        self.i = i
        self.sampling_rate = sampling_rate
        self.sweep_number = sweep_number

    def shift_time_by(self,time_steps):

        dt = 1. / self.sampling_rate
        self.t = self.t - time_steps*dt

    @property
    def t_end(self):
        return self.t[-1]


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
    def t_end(self):
        return self._prop('t_end')

    @property
    def sweep_number(self):
        return self._prop('sweep_number')
