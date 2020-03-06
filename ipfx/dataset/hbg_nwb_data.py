from typing import Dict, Any, List, Optional
import abc

from ipfx.stimulus import StimulusOntology
from ipfx.dataset.ephys_nwb_data import EphysNWBData, get_finite_or_none


class HBGNWBData(EphysNWBData):
    """
    Provides an Ephys Data Interface to an HBG generated NWB file

    """

    def __init__(self,
                 nwb_file: str,
                 ontology: StimulusOntology,
                 load_into_memory: bool = True,
                 validate_stim: bool = True,
                 ):
        super().init(nwb_file=nwb_file,
                     ontology=ontology,
                     load_into_memory=load_into_memory,
                     validate_stim=validate_stim)

    def get_stim_code_ext(self, sweep_number):
        return super().get_stim_code(sweep_number)

    def extract_sweep_record(self, sweep_num: int) -> Dict[str, Any]:
        attrs = self.nwb_data.get_sweep_attrs(sweep_num)

        sweep_record = {
            "sweep_number": sweep_num,
            "stimulus_units": self.get_stimulus_unit(sweep_num),
            "bridge_balance_mohm": get_finite_or_none(attrs, "bridge_balance"),
            "leak_pa": get_finite_or_none(attrs, "bias_current"),
            "stimulus_scale_factor": get_finite_or_none(attrs, "gain"),
            "stimulus_code": self.get_stim_code(sweep_num),
            "stimulus_code_ext": self.get_stim_code_ext(sweep_num)
        }

        if self.ontology:
            sweep_record["stimulus_name"] = self.get_stimulus_name(sweep_record["stimulus_code"])

        return sweep_record

