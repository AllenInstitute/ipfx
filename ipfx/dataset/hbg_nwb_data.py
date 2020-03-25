from typing import Dict, Any

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
                 validate_stim: bool = True
                 ):
        super(HBGNWBData, self).__init__(
            nwb_file=nwb_file,
            ontology=ontology,
            load_into_memory=load_into_memory,
            validate_stim=validate_stim
        )

    def get_stimulus_code_ext(self, sweep_number):
        return super().get_stimulus_code(sweep_number)

    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        attrs = self.get_sweep_attrs(sweep_number)

        sweep_record = {
            "sweep_number": sweep_number,
            "stimulus_units": self.get_stimulus_unit(sweep_number),
            "bridge_balance_mohm": get_finite_or_none(attrs, "bridge_balance"),
            "leak_pa": get_finite_or_none(attrs, "bias_current"),
            "stimulus_scale_factor": get_finite_or_none(attrs, "gain"),
            "stimulus_code": self.get_stimulus_code(sweep_number),
            "stimulus_code_ext": self.get_stimulus_code_ext(sweep_number),
            "clamp_mode": self.get_clamp_mode(sweep_number),
        }

        if self.ontology:
            sweep_record["stimulus_name"] = self.get_stimulus_name(
                sweep_record["stimulus_code"]
            )

        return sweep_record
