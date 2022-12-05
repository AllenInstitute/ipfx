import os
import logging
import warnings

import allensdk.core.json_utilities as ju

from enum import Enum


class StimulusType(Enum):
    RAMP = "ramp"
    LONG_SQUARE = "long_square"
    COARSE_LONG_SQUARE = "coarse_long_square"
    SHORT_SQUARE_TRIPLE = "short_square_triple"
    SHORT_SQUARE = "short_square"
    CHIRP = "chirp"
    SEARCH = "search"
    TEST = "test"
    BLOWOUT = "blowout"
    BATH = "bath"
    SEAL = "seal"
    BREAKIN = "breakin"
    EXTP = "extp"


STIMULUS_TYPE_NAME_MAPPING = {
    # Maps stimulus type to set of names
    StimulusType.RAMP: {"Ramp"},
    StimulusType.LONG_SQUARE: {
        "Long Square",
        "Long Square Threshold",
        "Long Square SupraThreshold",
        "Long Square SubThreshold",
    },
    StimulusType.COARSE_LONG_SQUARE: {
        "C1LSCOARSE",
    },
    StimulusType.SHORT_SQUARE_TRIPLE: {
        "Short Square - Triple",
    },
    StimulusType.SHORT_SQUARE: {
        "Short Square",
        "Short Square Threshold",
        "Short Square - Hold -60mV",
        "Short Square - Hold -70mV",
        "Short Square - Hold -80mV",
    },
    StimulusType.CHIRP: {
        "Chirp",
        "Chirp A Threshold",
        "Chirp B - Hold -65mV",
        "Chirp C - Hold -60mV",
        "Chirp D - Hold -55mV",
    },
    StimulusType.SEARCH: {"Search"},
    StimulusType.TEST: {"Test"},
    StimulusType.BLOWOUT: {"EXTPBLWOUT"},
    StimulusType.BATH: {"EXTPINBATH"},
    StimulusType.SEAL: {"EXTPCllATT"},
    StimulusType.BREAKIN: {"EXTPBREAKN"},
    StimulusType.EXTP: {"EXTP"}
}


def get_stimulus_type(stimulus_name):
    for stim_type, stim_names in STIMULUS_TYPE_NAME_MAPPING.items():
        if stimulus_name in stim_names:
            return stim_type
    else:
        raise ValueError(f"stimulus_name {stimulus_name} not found.\nSTIMULUS_TYPE_NAME_MAPPING: {STIMULUS_TYPE_NAME_MAPPING}")


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

        # Must match Stimulus Type Name Mapping, e.g 
        # for stimulus_type, names in _STIMULUS_TYPE_NAME_MAPPING.items():
        #     setattr(self, f"{stimulus_type.upper()}_NAMES", names)
        self.ramp_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.RAMP]
        self.long_square_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.LONG_SQUARE]
        self.coarse_long_square_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.COARSE_LONG_SQUARE]
        self.short_square_triple_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.SHORT_SQUARE_TRIPLE]
        self.short_square_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.SHORT_SQUARE]
        self.chirp_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.CHIRP]
        self.search_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.SEARCH]
        self.test_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.TEST]
        self.blowout_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.BLOWOUT]
        self.bath_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.BATH]
        self.seal_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.SEAL]
        self.breakin_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.BREAKIN]
        self.extp_names = STIMULUS_TYPE_NAME_MAPPING[StimulusType.EXTP]


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
