Refactor Handling of NWB data
=============================

Extracting ephys features from data requires constructing the EphysDatSet object that contains sweep traces 
as well as a sweep table with metadata. EphysDataSet owns the functionality for for handling sweep data data and metadata 
and should be agnostic of the data source.
 
Data from different sources (labs) may have somewhat different organization due to the loose nature of the NWB format or
because of using different version of NWB, extensions, etc. 

Thus, each data source may require a dedicated nwb data hangler class satisfying a given interface.
The nwb handlers for the for the AIBS data as well as for data from collaborators on the HBG grant are provided.


Some handy links:
- [design diagrams](http://confluence.corp.alleninstitute.org/pages/viewpage.action?spaceKey=PP&title=Publishing+IVSCC+NWB2+publish+module%2C+v3)
- [NWB2 DANDI icephys metadata list](https://docs.google.com/document/d/1KaQTtZ1AWSjHQsC3XO4k3BT5Z_pjVb4F6fIzrTEClhs/edit#heading=h.diqkrf5s0jti)
- [NWB2 icephys extension proposal](https://docs.google.com/document/d/1cAgsXv26BmQoVfa7Greyxs0oc4IGH-t5aJsm-AwUAAE/edit)

requirements
------------
1. NWB2 outputs are compatible with nwb-schema 2.2.1
1. NWB2 outputs are compatible with the [NWB icephys extension proposal](https://docs.google.com/document/d/1cAgsXv26BmQoVfa7Greyxs0oc4IGH-t5aJsm-AwUAAE/)
    - this should be implied by backwards compatibility of this proposal
1. multiple input NWB2 file types supported (particularly AIBS vs HBG; covered by dataset creation dynamic dispatch)
1. NWB1 file types version 1.0.5 is supported (for published data alreadly on the celltypes.brain-map.org)


inputs
------
1. An NWB2 or NWB1 file
1. Optional stimulus ontology, if not provided then use a default ontology
1. Optional sweep-level metadata. This is a mapping from sweep identifiers to dictionaries of metadata

outputs
-------
1. an nwb file of the same version as the input with attached spike times

approach
--------


```Python

from ipfx.sweep import Sweep,SweepSet
from typing import Dict, Any, Optional, List
import pandas as pd
from ipfx.nwb import MIESNWB1Data, MIESNWB2data, NWB2Data


class LabNotebook:
    def __init__(self):


class LabNotebookIgorNwb(LabNotebook):
    """
        Subclassing because this is corresponds to the existing implementation
    """
    def __init__(self,nwb_file: str):
        super(LabNotebook).__init__(self)


class NWBData:

    def __init__(self, nwb_file):


    def get_sweep_data(self, sweep_number):
        raise NotImplementedError

    def get_sweep_number(self, sweep_name):
        raise NotImplementedError

    def get_stim_code(self, sweep_number):
        raise NotImplementedError

    def get_spike_times(self, sweep_number):
        raise NotImplementedError

    def get_stimulus_code(self, sweep_number):
        raise NotImplementedError

    def get_stimulus_code(self, sweep_number):
        raise NotImplementedError

    def get_session_start_time(self):
        """
        Extract session_start_time in nwb
        Use last value if more than one is present

        Returns
        -------
        recording_date: str
            use date format "%Y-%m-%d %H:%M:%S", drop timezone info
        """


    def get_recording_date(self):


    def get_stimulus_unit(self, sweep_number):


    @staticmethod
    def get_unit_name(stim_attrs):


    @staticmethod
    def get_long_unit_name(unit):

    @staticmethod
    def validate_SI_unit(unit):



    def get_real_sweep_number(self, sweep_name, assumed_sweep_number=None):
        """
        Return the real sweep number for the given sweep_name. Falls back to
        assumed_sweep_number if given.
        """


    def get_starting_time(self, data_set_name):

    def get_sweep_attrs(self, sweep_number):


    def build_sweep_map(self):
        """
        Build table for mapping sweep_number to the names of stimulus and acquisition groups in the nwb file
        Returns
        -------
        """


    def drop_reacquired_sweeps(self):
        """
        If sweep was re-acquired, then drop earlier acquired sweep with the same sweep_number
        """

    def get_sweep_names(self):


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

    def get_acquisition_groups(self):


    def get_stimulus_groups(self):


    def add_spike_times(self, sweep_spike_times: Dict[int,List[float]]):
        """ Add spikes to the nwb file
        Parameters
        ----------
        sweep_spike_times: all detected spikes for the experiment
        """

    def add_metadata(self, metadata: Dict[str,Any]):
        """ Add metadata
        """

    def get_metadata(self, sweep_number: Optional[int] = None) -> Dict[str,Any]:
        """ Get experiment metadata
        If sweep_number is provided, then will return a sweep metadata, otherwise will return a cell-level metadata
        Parameters
        ----------
        sweep_number: sweep id
        
        Returns
        ------- 
        metadata
        
        """

    
class HBGNWB2Data(NWBData):
    def __init__(self, nwb_file: str:
        super.__init__(self, nwb_file)



class MIESNWB1Data(NWBData):
    def __init__(self, nwb_file: str):
        super.__init__(self, nwb_file)
        self.notebook = LabNotebookIgorNwb(nwb_file)



class MIESNWB2Data(NWBData):
    def __init__(self, nwb_file: str):
        super.__init__(self, nwb_file)
        self.notebook = LabNotebookIgorNwb(nwb_file)



def create_data_set(sweep_info = None,
                    nwb_file = None,
                    ontology = None,
                    api_sweeps = True,
                    validate_stim = True) -> EphysDataSet


    nwb_version = get_nwb_version(nwb_file)
    is_mies = is_file_mies(nwb_file)


    if nwb_version["major"] == 2 and is_mies:
        nwb_data = MIESNWB2Data(nwbfile)

    elif nwb_version["major"] == 2:
        nwb_data  = NWB2Data(nwb_file)

    elif nwb_version["major"] == 1:
        nwb_data = MIESNWB1Data(nwb_file)

    else:
        raise ValueError("Unsupported or unknown NWB major" +
                         "version {} ({})".format(nwb_version["major"], nwb_version["full"]))



    return EphysDataSet(sweep_info=sweep_info,
                       nwb_data=nwb_data,
                       ontology=ontology,
                       api_sweeps=api_sweeps,
                       validate_stim=validate_stim)


class EphysDataSet(object):

    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    STIMULUS_AMPLITUDE = 'stimulus_amplitude'
    STIMULUS_NAME = 'stimulus_name'
    SWEEP_NUMBER = 'sweep_number'
    CLAMP_MODE = 'clamp_mode'

    COLUMN_NAMES = [STIMULUS_UNITS,
                    STIMULUS_CODE,
                    STIMULUS_AMPLITUDE,
                    STIMULUS_NAME,
                    CLAMP_MODE,
                    SWEEP_NUMBER,
                    ]

    LONG_SQUARE = 'long_square'
    COARSE_LONG_SQUARE = 'coarse_long_square'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'
    SHORT_SQUARE = 'short_square'
    RAMP = 'ramp'

    VOLTAGE_CLAMP = "VoltageClamp"
    CURRENT_CLAMP = "CurrentClamp"

    def __init__(self,
        nwb_data: Union[MIESNWB2Data, NWB2Data, MIESNWB1Data],
        sweep_info: Optional[Dict[str,Any]] = None,
        ontology: Optional[Ontology] = None,
        api_sweeps: bool = True,
        validate_stim: bool = True):


    def build_sweep_table(self,
                          sweep_info: Optional[Dict[str,Any]] = None):
        """ Build sweep table including metadata used for filtering sweeps
        If sweep_info is not provided then will need to populate it

        Parameters
        ----------
        sweep_info: dict ot sweep metadata

        Returns
        -------

        """

    def filtered_sweep_table(self,
                             clamp_mode: Optional[str]=None,
                             stimuli: Optional[List[str]]=None,
                             stimuli_exclude=None,
                             ) -> pd.DataFrame:
        """Filter sweep table

        Parameters
        ----------
        clamp_mode: clamp mode self.VOLTAGE_CLAMP or self.CURRENT_CLAMP
        stimuli: stimuli to keep
        stimuli_exclude: stimuli to drop

        Returns
        -------
        filtered sweep table
        """


    def get_sweep_number(self,
                         stimuli: List[str],
                         clamp_mode: str=None) -> int:
        """Return sweep number

        Parameters
        ----------
        stimuli
        clamp_mode

        Returns
        -------
        sweep number:
        """

    def get_sweep_record(self, sweep_number: int) -> Dict[str,Any]:
        """
        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        sweep_record: dict of sweep properties
        """


    def sweep(self, sweep_number:int)-> Sweep:

        """
        Create an instance of the Sweep class with the data loaded from the from the nwb file

        Parameters
        ----------
        sweep_number: int

        Returns
        -------
        sweep: Sweep object
        """


    def sweep_set(self, sweep_numbers: List[int]) -> SweepSet:
        """ Create an instance of sweep set

        Parameters
        ----------
        sweep_numbers

        Returns
        -------

        """


    def get_sweep_data(self, sweep_number: int) -> Dict:
        """
        Read sweep data from the nwb file
        Substitute trailing zeros in the response for np.nan
        because trailing zeros occur after the end of recording

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



    def get_stimulus_name(self, stim_code: str) -> str:
        """ Get tne name of the stimulus based on stimulus code and validate stimulus code

        Parameters
        ----------
        stim_code

        Returns
        -------

        """

```



Testing
-------
t.b.d


Presentation
------------
t.b.d


Review Notes
------------

t.b.d
