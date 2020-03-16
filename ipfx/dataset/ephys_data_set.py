"""Interface required to run ipfx pipeline

"""
from typing import Optional, Sequence, List, Dict, Any
import abc

import pandas as pd

from ipfx.dataset.stimulus import StimulusOntology
from ipfx.dataset.ephys_data_interface import EphysDataInterface
from ipfx.sweep import Sweep, SweepSet


class NotYetImplementedError(TypeError):
    """This method or property is not yet implemented. Unlike 
    NotImplementedError, it will be implemented in the future!
    """


class _EphysDataset(abc.ABC):
    """Interface expected by pipeline modules, specifically:
        - sweep extraction
        - auto qc
        - feature extraction

    Provides data, but does not directly read from any external sources (e.g. 
    NWB files). Instead, an EphysDataInterface is used as an intermediary, 
    handling all of the gory business of extracting 

    """

    @abc.abstractproperty
    def ontology(self) -> StimulusOntology:
        """Used to interpret the stimuli presented during this experiment
        """

    @abc.abstractproperty
    def sweep_info(self) -> List[Dict[str, Any]]:
        """Each element is a dictionary describing a sweep. See also 
        sweep_table.
        """

    @abc.abstractproperty
    def sweep_table(self) -> pd.DataFrame:
        """Each row describes a sweep. Rows are ordered by presentation time. 
        Some datasets may provide additional information on their sweeps, but 
        all have at least:
            stimulus_units
            stimulus_code
            stimulus_ampliture
            stimulus_name
            sweep_number
            clamp_mode
        """

    @abc.abstractmethod
    def __init__(self, data: EphysDataInterface):
        """Construct an EphysPipelineDataset.

        Parameters
        ----------
        data : This EphysDataInterface will handle the details of loading data
            from an external source.
        """

    def filtered_sweep_table(
            self,
            clamp_mode: Optional[str] = None,
            stimuli: Optional[Sequence[str]] = None,
            stimuli_exclude: Optional[str] = None
    ) -> pd.DataFrame:
        """Convenvience for filtering the sweep table

        Parameters
        ----------
        clamp_mode : restrict to one of:
                VoltageClamp
                CurrentClamp
            sweeps or leave None for both.
        stimuli : include only stimuli with these names
        stimuli_exclude : do not include any stimuli with these names
        """

    def sweep(self, sweep_number: int) -> Sweep:
        """Obtain a Sweep object containing data for a specified sweep

        Parameters
        ----------
        sweep_number: identifier of the sweep to be accessed.

        Returns
        -------
        Data and metadata for a single sweep
        """

    def sweep_set(self, sweep_numbers: Sequence[int]) -> SweepSet:
        """Obtain a SweepSet object, which holds data for a collection of 
        sweeps.
        
        Parameters
        ----------
        sweep_numbers : identifiers for the sweeps which will make up the set

        Returns
        -------
        Data and metadata for a collection of sweeps
        """

    def get_stimulus_name(self, stim_code: str) -> Optional[str]:
        """Convenience for looking up a stimulus' name from its code. See also 
        Ontology.get_stimulus_name
        
        Parameters
        ----------
        stim_code : The code to look up

        Returns
        -------
        The name of the stimulus, or None if it was not found
        """

    def get_stimulus_code(self, sweep_number: int) -> str:
        """Look up the stimulus code for a specific sweep

        Parameters
        ----------
        sweep_number : Identifies sweep whose code will be looked up

        Returns
        -------
        The code of the stimulus applied on the identified sweep
        """

    def get_stimulus_code_ext(self, sweep_number:)