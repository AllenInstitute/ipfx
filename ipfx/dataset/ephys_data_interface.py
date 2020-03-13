from typing import Dict, Any, List, Optional
import abc
import warnings
from ipfx.stimulus import StimulusOntology

class EphysDataInterface(abc.ABC):
    """
    The interface that any child class providing data to the EphysDataSet must implement

    """

    def __init__(self, ontology: StimulusOntology):

        self.ontology = ontology

    @abc.abstractmethod
    def get_sweep_data(self, sweep_number: int) -> Dict[str,Any]:
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
    def get_sweep_record(self, sweep_number: int) -> Dict[str,Any]:
        """
        Extract sweep data

        Parameters
        ----------
        sweep_number

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
    def get_sweep_number(self, sweep_name:str)-> int:
        """
        Infer sweep number from the sweep_name

        Parameters
        ----------
        sweep_name

        Returns
        -------

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_stim_code(self, sweep_number: int) -> str:
        """
        Extract stimulus code

        Parameters
        ----------
        sweep_number

        Returns
        -------
        stimulus code
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_stimulus_name(self,
                          stim_code: str,
                          validate: Optional[bool] = True) -> str:
        """
        Extract name of the stimulus from the stimulus given the ontology

        Parameters
        ----------
        validate: flag to validate the stimulus code is in the ontology

        Returns
        -------
        stimulus name
        """

    @abc.abstractmethod
    def get_stim_code_ext(self, sweep_number: int)-> str:
        """
        Extract stimulus code with the extension of the format: stim_code + %d

        Parameters
        ----------
        sweep_number

        Returns
        -------
        stimulus code with extension
        """
        raise NotImplementedError


    @abc.abstractmethod
    def get_session_start_time(self) -> str:
        """
        Extract session_start_time in nwb
        Use last value if more than one is present

        Returns
        -------
        recording_date: str
            use date format "%Y-%m-%d %H:%M:%S", drop timezone info
        """

        raise NotImplementedError

    @abc.abstractmethod
    def get_recording_date(self) -> str:
        """
        Extract recording date

        Returns
        -------
        recording date
        """

        raise NotImplementedError


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
    def get_real_sweep_number(self, sweep_name:str, assumed_sweep_number: Optional[int]=None) -> int:
        """
        Return the real sweep number for the given sweep_name. Falls back to
        assumed_sweep_number if given.

        Parameters
        ----------
        sweep_name
        assumed_sweep_number

        Returns
        -------
        sweep number

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
    def get_acquisition_groups(self) -> List[str]:
        """
        Collect names of hdf5 groups from the acquisition

        Returns
        -------
        names of acquisition groups
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
