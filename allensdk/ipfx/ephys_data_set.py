import json
import os
import logging

DEFAULT_STIMULUS_ONTOLOGY_FILE = os.path.join(os.path.dirname(__file__), 'stimulus_ontology.json')

def load_default_stimulus_ontology():
    logging.debug("loading default stimulus ontology: %s", DEFAULT_STIMULUS_ONTOLOGY_FILE)
    with open(DEFAULT_STIMULUS_ONTOLOGY_FILE) as f:
        return StimulusOntology(json.load(f))


class Stimulus(object):

    def __init__(self, tag_sets):
        self.tag_sets = tag_sets

    def tags(self, tag_type=None, flat=False):

        tag_sets = self.tag_sets
        if tag_type:
            tag_sets = [ ts for ts in tag_sets if ts[0] == tag_type ]
        if flat:
            return [ t for tag_set in tag_sets for t in tag_set ]
        else:
            return tag_sets

    def has_tag(self, tag, tag_type=None):
        return tag in self.tags(tag_type=tag_type, flat=True)


class StimulusOntology(object):

    """

    Creates stimuli based on stimulus ontology
    """

    def __init__(self, stimuli):

        """

        Parameters
        ----------
            stimuli: nested list
                Stimuli ontology

        """

        self.stimuli = list(Stimulus(s) for s in stimuli)

    def find(self, tag, tag_type=None):

        matching_stims = [ s for s in self.stimuli if s.has_tag(tag, tag_type=tag_type) ]

        if not matching_stims:
            raise KeyError("Could not find stimulus: %s" % tag)

        return matching_stims

    def find_one(self, tag, tag_type=None):
        matching_stims = self.find(tag, tag_type)

        if len(matching_stims) > 1:
            raise KeyError("Multiple stimuli match '%s', one expected" % tag)

        return matching_stims[0]

    def stimulus_has_any_tags(self, stim, tags, tag_type=None):
        matching_stim = self.find_one(stim, tag_type)
        return any(matching_stim.has_tag(t) for t in tags)

    def stimulus_has_all_tags(self, stim, tags, tag_type=None):
        matching_stim = self.find_one(stim, tag_type)
        return all(matching_stim.has_tag(t) for t in tags)


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

    def __init__(self, ontology=None):
        self.sweep_table = None

        if ontology is None:
            ontology = load_default_stimulus_ontology()

        self.ontology = ontology

        self.ramp_names = ( "Ramp", )

        self.long_square_names = ( "Long Square", )
        self.coarse_long_square_names = ( "C1LSCOARSE",  )
        self.short_square_triple_names = ( "Short Square - Triple", )

        self.short_square_names = ( "Short Square",
                                    "Short Square - Hold -60mV",
                                    "Short Square - Hold -70mV",
                                    "Short Square - Hold -80mV" )

        self.blowout_names = ( 'EXTPBLWOUT', )
        self.bath_names = ( 'EXTPINBATH', )
        self.seal_names = ( 'EXTPCllATT', )
        self.breakin_names = ( 'EXTPBREAKN', )
        self.extp_names = ( 'EXTP', )


        self.current_clamp_units = ( 'Amps', 'pA')


    def filtered_sweep_table(self, current_clamp_only=False, passing_only=False, stimuli=None):

        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(self.current_clamp_units)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(self.ontology.stimulus_has_any_tags, args=(stimuli,))
            st = st[mask]

        return st

    def sweep(self, sweep_number):
        """ returns a dictionary with properties: i (in pA), v (in mV), t (in sec), start, end"""
        raise NotImplementedError

    def sweep_set(self, sweep_numbers):
        return SweepSet([ self.sweep(sn) for sn in sweep_numbers ])

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        pass

class Sweep(object):
    def __init__(self, t, v, i, expt_start=None, expt_end=None, sampling_rate=None):
        self.t = t
        self.v = v
        self.i = i

        self.expt_start = expt_start if expt_start else 0
        self.expt_end = expt_end if expt_end else self.t_end
        self.sampling_rate = sampling_rate

    @property
    def t_end(self):
        return self.t[-1]


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
    def t_end(self):
        return self._prop('t_end')
