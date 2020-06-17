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
            "cap_neutralization": get_finite_or_none(attrs, "capacitance_compensation"),
            # we could use get_finite_or_none(attrs, "capacitance_fast") here
            "cp_fast_cap": self.notebook.get_value(
                "Fast compensation capacitance", sweep_number, None
            ),
            "cp_fast_tau": self.notebook.get_value(
                "Fast compensation time", sweep_number, None
            ),
            # we could use get_finite_or_none(attrs, "capacitance_slow") here
            "cp_slow_cap": self.notebook.get_value(
                "Slow compensation capacitance", sweep_number, None
            ),
            "cp_slow_tau": self.notebook.get_value(
                "Slow compensation time", sweep_number, None
            ),
            "rs_comp_bandwidth": get_finite_or_none(attrs, "resistance_comp_bandwidth"),
            "rs_comp_correction": get_finite_or_none(attrs, "resistance_comp_correction"),
            "rs_comp_prediction": get_finite_or_none(attrs, "resistance_comp_prediction"),
            "whole_cell_cap_comp": get_finite_or_none(attrs, "whole_cell_capacitance_comp"),
            "whole_cell_sr_comp": get_finite_or_none(attrs, "whole_cell_series_resistance_comp"),
            "stimulus_scale_factor": self.notebook.get_value(
                "Scale Factor", sweep_number, None
            ),
            "vclamp_holding_level": self.notebook.get_value(
                "V-Clamp Holding Level", sweep_number, None
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
