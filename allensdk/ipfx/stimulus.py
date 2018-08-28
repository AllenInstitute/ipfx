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
        stimuli: nested list  of stimuli ontology properties

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

