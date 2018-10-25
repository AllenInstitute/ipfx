import logging


class EphysDataSet(object):
    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    STIMULUS_AMPLITUDE = 'stimulus_amplitude'
    STIMULUS_NAME = 'stimulus_name'
    SWEEP_NUMBER = 'sweep_number'
    PASSED = 'passed'

    LONG_SQUARE = 'long_square'
    COARSE_LONG_SQUARE = 'coarse_long_square'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'
    SHORT_SQUARE= 'short_square'
    RAMP = 'ramp'

    def __init__(self, ontology):
        self.sweep_table = None

        self.ontology = ontology

    def filtered_sweep_table(self,
                             current_clamp_only=False,
                             passing_only=False,
                             stimuli=None,
                             exclude_search=False,
                             exclude_test=False,
                             exclude_truncated=False,
                             ):

        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(self.ontology.current_clamp_units)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(self.ontology.stimulus_has_any_tags, args=(stimuli,), tag_type="code")
            st = st[mask]

        if exclude_search:
            mask = ~st[self.STIMULUS_NAME].isin(self.ontology.search_names)
            st = st[mask]

        if exclude_test:
            mask = ~st[self.STIMULUS_NAME].isin(self.ontology.test_names)
            st = st[mask]

        if exclude_truncated:
            mask = ~(st["truncated"] == True)
            st = st[mask]

        return st

    def get_sweep_number_by_stimulus_names(self, stimulus_names):

        sweeps = self.filtered_sweep_table(stimuli=stimulus_names).sort_values(by='sweep_number')

        if len(sweeps) > 1:
            logging.warning("Found multiple sweeps for stimulus %s: using largest sweep number" % str(stimulus_names))

        if len(sweeps) == 0:
            raise IndexError

        return sweeps.sweep_number.values[-1]

    def get_sweep_info_by_sweep_number(self, sweep_number):
        """

        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        sweep_info: dict of sweep properties
        """
        st = self.sweep_table
        mask = st[self.SWEEP_NUMBER] == sweep_number
        st = st[mask]

        return st.to_dict(orient='records')[0]


    def sweep(self, sweep_number):
        """ returns a dictionary with properties: i (in pA), v (in mV), t (in sec), start, end"""
        raise NotImplementedError

    def sweep_set(self, sweep_numbers):
        return SweepSet([ self.sweep(sn) for sn in sweep_numbers ])

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        pass


class Sweep(object):
    def __init__(self, t, v, i, expt_idx_range, sampling_rate=None, id=None, clamp_mode=None):
        self.t = t
        self.v = v
        self.i = i
        self.expt_idx_range = expt_idx_range
        self.sampling_rate = sampling_rate
        self.id = id
        self.clamp_mode = clamp_mode

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
        return [ getattr(s, prop) for s in self.sweeps ]

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
    def id(self):
        return self._prop('id')
