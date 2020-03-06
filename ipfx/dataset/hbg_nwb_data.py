from typing import Dict, Any, List, Optional
import abc

from ipfx.stimulus import StimulusOntology
from ipfx.dataset.ephys_nwb_data import EphysNWBData


class HBGNWBData(EphysNWBData):
    """
    Provides an Ephys Data Interface to an HBG generated NWB file

    """

    def __init__(self,
                 nwb_file: str,
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 ):
        super().init(nwb_file=nwb_file,
                     ontology=ontology,
                     load_into_memory=load_into_memory)

    def get_stim_code_ext(self, sweep_number):
        return super().get_stim_code(sweep_number)

