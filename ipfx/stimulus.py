import os
import logging
import warnings

import allensdk.core.json_utilities as ju



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

    DEFAULT_STIMULUS_ONTOLOGY_FILE = os.path.join(
        os.path.dirname(__file__), 
        "defaults",
        "stimulus_ontology.json"
    )

    def __init__(self, stim_ontology_tags=None):

        """

        Parameters
        ----------
        stim_ontology_tags: nested list  of stimuli ontology properties

        """

        self.stimuli = list(Stimulus(s) for s in stim_ontology_tags)

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

    def find(self, tag, tag_type=None):
        """
        Find stimuli matching a given tag
        Parameters
        ----------
        tag: str
        tag_type: str

        Returns
        -------
        matching_stims: list of Stimuli objects

        """
        matching_stims = [ s for s in self.stimuli if s.has_tag(tag, tag_type=tag_type) ]

        if not matching_stims:
            warnings.warn("Could not find stimulus: %s" % tag)
            matching_stims = [Stimulus([["code", "unknown"], ["name", "Unknown"]])]

        return matching_stims

    def find_one(self, tag, tag_type=None):
        matching_stims = self.find(tag, tag_type)

        if len(matching_stims) > 1:
            warnings.warn("Multiple stimuli match '%s', one expected" % tag)
        return matching_stims[0]

    def stimulus_has_any_tags(self, stim, tags, tag_type=None):
        """
        Find stimulus based on a tag stim and then check if it has any tags
        Parameters
        ----------
        stim: str
            tag to find stimulus
        tags: str
            tags to check in any belong to the stimulus
        tag_type

        Returns
        -------
        bool: True if any tags match, otherwise False
        """
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

    @classmethod
    def default(cls):
        """Construct an ontology object using default tags
        """
        ontology_data = ju.read(cls.DEFAULT_STIMULUS_ONTOLOGY_FILE)
        return cls(ontology_data)
