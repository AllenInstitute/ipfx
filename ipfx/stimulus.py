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

    def __init__(self, stimuli_props):

        """

        Parameters
        ----------
        stimuli: nested list  of stimuli ontology properties

        """
        self.ramp_names = ( "Ramp",)

        self.long_square_names = ( "Long Square",
                                   "Long Square Threshold",
                                   "Long Square SupraThreshold",
                                   "Long Square SubThreshold" )

        self.coarse_long_square_names = ( "C1LSCOARSE",)
        self.short_square_triple_names = ( "Short Square - Triple", )

        self.short_square_names = ( "Short Square",
                                    "Short Square Threshold",
                                    "Short Square - Hold -60mV",
                                    "Short Square - Hold -70mV",
                                    "Short Square - Hold -80mV" )

        self.search_names = ("Search",)
        self.test_names = ("Test",)
        self.blowout_names = ( 'EXTPBLWOUT', )
        self.bath_names = ( 'EXTPINBATH', )
        self.seal_names = ( 'EXTPCllATT', )
        self.breakin_names = ( 'EXTPBREAKN', )
        self.extp_names = ( 'EXTP', )

        self.current_clamp_units = ( 'Amps', 'pA')



        self.stimuli = list(Stimulus(s) for s in stimuli_props)

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
        matching_stim = self.find(stim, tag_type)

        if len(matching_stim) > 1:
            logging.warning("Found multiple stimuli with the tag: %s" % stim)

        matching_tags = []

        for st in matching_stim:
            for t in tags:
                matching_tags.append(st.has_tag(t))

        return any(matching_tags)

    def stimulus_has_all_tags(self, stim, tags, tag_type=None):
        matching_stim = self.find_one(stim, tag_type)
        return all(matching_stim.has_tag(t) for t in tags)

