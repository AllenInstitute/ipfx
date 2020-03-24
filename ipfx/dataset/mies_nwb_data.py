from typing import Dict, Any

from ipfx.stimulus import StimulusOntology
from ipfx.dataset.labnotebook import LabNotebookReader
from ipfx.dataset.ephys_nwb_data import EphysNWBData, get_finite_or_none


class MIESNWBData(EphysNWBData):
    """
    Provides an Ephys Data Interface to a MIES generated NWB file

    """

    def __init__(
            self,
            nwb_file: str,
            notebook: LabNotebookReader,
            ontology: StimulusOntology,
            load_into_memory: bool = True,
            validate_stim: bool = True
    ):
        super(MIESNWBData, self).__init__(
            nwb_file=nwb_file,
            ontology=ontology,
            load_into_memory=load_into_memory,
            validate_stim=validate_stim
        )
        self.notebook = notebook

    def get_stim_code_ext(self, sweep_number):
        stim_code = super().get_stimulus_code(sweep_number)

        cnt = self.notebook.get_value("Set Sweep Count", sweep_number, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)
        return stim_code_ext

    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        attrs = self.get_sweep_attrs(sweep_number)

        sweep_record = {
            "sweep_number": sweep_number,
            "stimulus_units": self.get_stimulus_unit(sweep_number),
            "bridge_balance_mohm": get_finite_or_none(attrs, "bridge_balance"),
            "leak_pa": get_finite_or_none(attrs, "bias_current"),
            "stimulus_scale_factor": self.notebook.get_value(
                "Scale Factor", sweep_number, None
            ),
            "stimulus_code": self.get_stimulus_code(sweep_number),
            "stimulus_code_ext": self.get_stim_code_ext(sweep_number),
            "clamp_mode": self.get_clamp_mode(sweep_number)
        }

        if self.ontology:
            sweep_record["stimulus_name"] = self.get_stimulus_name(
                sweep_record["stimulus_code"]
            )

        return sweep_record
