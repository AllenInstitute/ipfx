from typing import Dict, Any, List, Optional
import abc

from ipfx.stimulus import StimulusOntology
from ipfx.dataset.lab_notebook_reader import LabNotebookReader
from ipfx.dataset.ephys_nwb_data import EphysNWBData


class MIESNWBData(EphysNWBData):
    """
    Provides an Ephys Data Interface to a MIES generated NWB file

    """

    def __init__(self,
                 nwb_file: str,
                 lab_notebook_reader: LabNotebookReader
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 ):
        super().init(nwb_file=nwb_file,
                     ontology=ontology,
                     load_into_memory=load_into_memory)
        self.notebook = lab_notebook_reader

    def get_stim_code_ext(self, sweep_number):
        stim_code = super().get_stim_code(sweep_number)

        cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)
        return stim_code_ext