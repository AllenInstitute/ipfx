import logging
import re

import numpy as np
from ipfx.stimulus import StimulusOntology
import ipfx.epochs as ep


class EphysDataSet(object):

    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    STIMULUS_AMPLITUDE = 'stimulus_amplitude'
    STIMULUS_NAME = 'stimulus_name'
    SWEEP_NUMBER = 'sweep_number'
    PASSED = 'passed'
    CLAMP_MODE = 'clamp_mode'

    LONG_SQUARE = 'long_square'
    COARSE_LONG_SQUARE = 'coarse_long_square'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'
    SHORT_SQUARE = 'short_square'
    RAMP = 'ramp'

    def __init__(self, ontology=None):
        self.sweep_table = None
        self.ontology = ontology if ontology else StimulusOntology()

    def filtered_sweep_table(self,
                             current_clamp_only=False,
                             passing_only=False,
                             stimuli=None,
                             exclude_search=False,
                             exclude_test=False,
                             sweep_number=None,
                             ):

        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(
                self.ontology.current_clamp_units)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, args=(stimuli,), tag_type="code")
            st = st[mask]

        if exclude_search:
            mask = ~st[self.STIMULUS_NAME].isin(self.ontology.search_names)
            st = st[mask]

        if exclude_test:
            mask = ~st[self.STIMULUS_NAME].isin(self.ontology.test_names)
            st = st[mask]

        if sweep_number is not None:
            mask = st[self.SWEEP_NUMBER] == sweep_number
            st = st[mask]

        return st

    def get_sweep_number_by_stimulus_names(self, stimulus_names):

        sweeps = self.filtered_sweep_table(
            stimuli=stimulus_names).sort_values(by='sweep_number')

        if len(sweeps) > 1:
            logging.warning(
                "Found multiple sweeps for stimulus %s: using largest sweep number" % str(stimulus_names))

        if len(sweeps) == 0:
            raise IndexError

        return sweeps.sweep_number.values[-1]

    def get_sweep_meta_data(self, sweep_number):
        """

        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        sweep_meta_data: dict of sweep properties
        """

        sweep_meta_data = self.filtered_sweep_table(sweep_number=sweep_number)

        return sweep_meta_data.to_dict(orient='records')[0]

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

        sweep_data = self.get_sweep_data(sweep_number)
        sweep_meta_data = self.get_sweep_meta_data(sweep_number)
        sampling_rate = sweep_data['sampling_rate']
        dt = 1. / sampling_rate
        sweep_start_ix, sweep_end_ix = ep.get_sweep_epoch(sweep_data['response'])

        response = sweep_data['response'][sweep_start_ix:sweep_end_ix+1]
        stimulus = sweep_data['stimulus'][sweep_start_ix:sweep_end_ix+1]
        t = np.arange(sweep_start_ix, sweep_end_ix + 1) * dt

        if sweep_meta_data[self.CLAMP_MODE] == "VoltageClamp":
            v = stimulus
            i = response
            expt_idx_range = None
        elif sweep_meta_data[self.CLAMP_MODE] == "CurrentClamp":
            v = response
            i = stimulus
            expt_idx_range = ep.get_experiment_epoch(stimulus, response, sampling_rate)
            expt_start_ix,expt_end_ix = expt_idx_range
            t = t - expt_start_ix*dt
        else:
            raise Exception("Unable to determine clamp mode for sweep " + sweep_number)

        try:
            sweep = Sweep(t=t,
                          v=v,
                          i=i,
                          sampling_rate=sampling_rate,
                          expt_idx_range=expt_idx_range,
                          sweep_number=sweep_number,
                          )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep

    def sweep_set(self, sweep_numbers):
        try:
            return SweepSet([self.sweep(sn) for sn in sweep_numbers])
        except TypeError:  # not iterable
            return SweepSet([self.sweep(sweep_numbers)])

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        raise NotImplementedError

    def extract_sweep_meta_data(self):
        """
        Returns
        -------
        sweep_props: list of dicts
            where each dict includes sweep properties
        """

        raise NotImplementedError

    def modify_api_sweep_info(self, sweep_list):
        return [{EphysDataSet.SWEEP_NUMBER: s['sweep_number'],
                 EphysDataSet.STIMULUS_UNITS: s['stimulus_units'],
                 EphysDataSet.STIMULUS_AMPLITUDE: s['stimulus_absolute_amplitude'],
                 EphysDataSet.STIMULUS_CODE: re.sub(r"\[\d+\]", "", s['stimulus_description']),
                 EphysDataSet.STIMULUS_NAME: s['stimulus_name'],
                 EphysDataSet.PASSED: True} for s in sweep_list]

    def get_sweep_data(self, sweep_number):
        """
        Return the data of sweep_number as dict. The dict has the format:

        {
            'stimulus': np.ndarray,
            'response': np.ndarray,
            'stimulus_unit': string,
            'sampling_rate': float
        }

        """
        raise NotImplementedError

    def get_stimulus_name(self, stim_code):

        if not self.ontology:
            raise ValueError("Missing stimulus ontology")

        stim = self.ontology.find_one(stim_code, tag_type="code")
        return stim.tags(tag_type="name")[0][-1]


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
