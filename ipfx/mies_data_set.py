from typing import Dict, Any, Optional

from ipfx.hbg_dataset import HBGDataSet, get_finite_or_none
from ipfx import lab_notebook_reader

class MiesDataSet(HBGDataSet):

    def __init__(self, sweep_info=None, nwb_file=None, ontology=None, api_sweeps=True, validate_stim=True):
        self.notebook = lab_notebook_reader.create_lab_notebook_reader(nwb_file)

        super(MiesDataSet, self).__init__(
            sweep_info=sweep_info,
            nwb_file=nwb_file,
            ontology=ontology,
            api_sweeps=api_sweeps,
            validate_stim=validate_stim
        )

    def get_stimulus_code_ext(self, sweep_num):

        stim_code = self.nwb_data.get_stim_code(sweep_num)

        cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)
        print(stim_code_ext)
        return stim_code_ext

    def extract_sweep_record(self, sweep_num: int) -> Dict[str, Any]:
        attrs = self.nwb_data.get_sweep_attrs(sweep_num)

        sweep_record = {
            "sweep_number": sweep_num,
            "stimulus_units": self.get_stimulus_units(sweep_num),
            "bridge_balance_mohm": get_finite_or_none(attrs, "bridge_balance"),
            "leak_pa": get_finite_or_none(attrs, "bias_current"),
            "stimulus_scale_factor": self.notebook.get_value("Scale Factor", sweep_num, None),
            "stimulus_code": self.get_stimulus_code(sweep_num),
            "stimulus_code_ext": self.get_stimulus_code_ext(sweep_num)
        }

        if self.ontology:
            sweep_record["stimulus_name"] = self.get_stimulus_name(sweep_record["stimulus_code"])

        return sweep_record
