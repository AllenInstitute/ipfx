import ipfx.epochs as ep


class Sweep(object):
    def __init__(self, t, v, i, clamp_mode, sampling_rate, sweep_number=None, epochs=None):
        self._t = t
        self._v = v
        self._i = i
        self.sampling_rate = sampling_rate
        self.sweep_number = sweep_number
        self.clamp_mode = clamp_mode
        if epochs:
            self.epochs = epochs
        else:
            self.epochs = {}

        if self.clamp_mode == "CurrentClamp":
            self._response = self._v
            self._stimulus = self._i
        else:
            self._response = self._i
            self._stimulus = self._v

        self.detect_epochs()
        self.selected_epoch_name = "recording"

    @property
    def t(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._t[start_idx:end_idx+1]

    @property
    def v(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._v[start_idx:end_idx+1]

    @property
    def i(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._i[start_idx:end_idx+1]

    def select_epoch(self, epoch_name):
        self.selected_epoch_name = epoch_name

    def set_time_zero_to_index(self, time_step):
        dt = 1. / self.sampling_rate
        self._t = self._t - time_step*dt

    def detect_epochs(self):
        """
        Detect epochs if they are not provided in the constructor

        """

        if "test" not in self.epochs:
            self.epochs["test"] = ep.get_test_epoch(self._stimulus, self.sampling_rate)
        if self.epochs["test"]:
            test_pulse = True
        else:
            test_pulse = False

        if "sweep" not in self.epochs:
            self.epochs["sweep"] = ep.get_sweep_epoch(self._i)
        if "recording" not in self.epochs:
            self.epochs["recording"] = ep.get_recording_epoch(self._response)
        # get valid recording by selecting epoch and using i/v prop before detecting stim
        self.select_epoch("recording")
        stim = self.i if self.clamp_mode == "CurrentClamp" else self.v
        if "stim" not in self.epochs:
            self.epochs["stim"] = ep.get_stim_epoch(stim, test_pulse)
        if "experiment" not in self.epochs:
            self.epochs["experiment"] = ep.get_experiment_epoch(stim, self.sampling_rate, test_pulse)


class SweepSet(object):
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def _prop(self, prop):
        return [getattr(s, prop) for s in self.sweeps]

    def select_epoch(self, epoch_name):
        for sweep in self.sweeps:
            sweep.select_epoch(epoch_name)

    def align_to_start_of_epoch(self, epoch_name):

        for sweep in self.sweeps:
            start_idx, end_idx = sweep.epochs[epoch_name]
            sweep.set_time_zero_to_index(start_idx)

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
    def sweep_number(self):
        return self._prop('sweep_number')

    @property
    def sampling_rate(self):
        return self._prop('sampling_rate')
