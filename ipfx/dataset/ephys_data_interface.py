from typing import Dict, Any, List, Optional, Sequence
import abc
import warnings
from datetime import datetime

from ipfx.stimulus import StimulusOntology

class EphysDataInterface(abc.ABC):
    """
    The interface that any child class providing data to the EphysDataSet must implement

    """

    def __init__(self, ontology: StimulusOntology):

        self.ontology = ontology

    @abc.abstractproperty
    def sweep_numbers(self) -> Sequence[int]:
        """A time-ordered sequence of each sweep's integer identifer
        """

    @abc.abstractmethod
    def get_sweep_data(self, sweep_number: int) -> Dict[str, Any]:
        """
        Extract sweep data

        Parameters
        ----------
        sweep_number

        Returns
        -------

        dict in the format:

        {
            'stimulus': np.ndarray,
            'response': np.ndarray,
            'stimulus_unit': string,
            'sampling_rate': float
        }

        """
        raise NotImplementedError


    @abc.abstractmethod
    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        """Returns metadata about a sweep

        Parameters
        ----------
        sweep_number : identifier of the sweep whose metadata will be returned

        Returns
        -------

        dict in the format:

        {
            "sweep_number": int,
            "stimulus_units": str,
            "bridge_balance_mohm": float,
            "leak_pa": float,
            "stimulus_scale_factor": float,
            "stimulus_code": str,
            "stimulus_code_ext": str,
            "stimulus_name": str,
            "clamp_mode": str
        }

        """
        raise NotImplementedError


    @abc.abstractmethod
    def get_sweep_attrs(self, sweep_number) -> Dict[str,Any]:
        """
        Extract sweep attributes

        Parameters
        ----------
        sweep_number

        Returns
        -------
        sweep attributes
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_stimulus_code(self, sweep_number: int) -> str:
        """Obtain the code of the stimulus presented on a particular sweep.

        Parameters
        ----------
        sweep_number : unique identifier for the sweep

        Returns
        -------
        The code (a short-form description) of the stimulus presented on the 
        identified sweep
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_full_recording_date(self) -> datetime:
        """Obtain the full date and time at which recording began.

        Returns
        -------
        A datetime object, with timezone, reporting the start of recording
        """

    @abc.abstractmethod
    def get_stimulus_unit(self, sweep_number: int) -> str:
        """
        Extract unit of a stimulus

        Parameters
        ----------
        sweep_number

        Returns
        -------
        stimulus unit
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_starting_time(self, data_set_name: str) -> str:
        """

        Parameters
        ----------
        data_set_name

        Returns
        -------
        starting time of acquisition


        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_sweep_map(self):
        """
        Build table for mapping sweep_number to the names of stimulus and acquisition groups in the nwb file
        Returns
        -------
        """

        raise NotImplementedError

    @abc.abstractmethod
    def drop_reacquired_sweeps(self):
        """
        If sweep was re-acquired, then drop earlier acquired sweep with the same sweep_number
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_sweep_map(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int
            real sweep number
        Returns
        -------
        sweep_map: dict
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_stimulus_groups(self) ->List[str]:
        """
        Collect names of hdf5 groups from the stimulus

        Returns
        -------
        names of acquisition groups
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
