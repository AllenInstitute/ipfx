import warnings
import logging
import re

import numpy as np
from ipfx.stimulus import StimulusOntology
from ipfx.sweep import Sweep,SweepSet


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

    current_clamp_units = ('Amps', 'pA')

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
            st = st[st[self.STIMULUS_UNITS].isin(self.current_clamp_units)]

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
        Create an instance of the Sweep class with the data loaded from the from a file

        Parameters
        ----------
        sweep_number: int

        Returns
        -------
        sweep: Sweep object
        """

        sweep_data = self.get_sweep_data(sweep_number)
        sweep_meta_data = self.get_sweep_meta_data(sweep_number)
        sampling_rate = sweep_data['sampling_rate']
        dt = 1. / sampling_rate
        t = np.arange(0, len(sweep_data['stimulus'])) * dt

        epochs = sweep_data.get('epochs')
        clamp_mode = self.get_clamp_mode(sweep_meta_data['stimulus_units'])

        if clamp_mode == "VoltageClamp":
            v = sweep_data['stimulus']
            i = sweep_data['response']
        elif clamp_mode == "CurrentClamp":
            v = sweep_data['response']
            i = sweep_data['stimulus']
        else:
            raise Exception("Unable to determine clamp mode for sweep " + sweep_number)

        try:
            sweep = Sweep(t=t,
                          v=v,
                          i=i,
                          sampling_rate=sampling_rate,
                          sweep_number=sweep_number,
                          clamp_mode=clamp_mode,
                          epochs=epochs,
                          )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep

    def get_clamp_mode(self, stimulus_unit):

        clamp_mode = "CurrentClamp" if stimulus_unit in self.current_clamp_units else "VoltageClamp"

        return clamp_mode

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

        try:
            stim = self.ontology.find_one(stim_code, tag_type="code")
            return stim.tags(tag_type="name")[0][-1]

        except KeyError:
            if self.validate_stim:
                raise
            else:
                warnings.warn("Stimulus code {} is not in the ontology".format(stim_code))
                return
