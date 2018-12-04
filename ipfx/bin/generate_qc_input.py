import os
import allensdk.core.json_utilities as ju
from ipfx._schemas import GeneratePipelineInputParameters
import argschema as ags
from run_sweep_extraction import run_sweep_extraction
from generate_se_input import generate_se_input
import ipfx.sweep_props as sp

QC_INPUT_FEATURES = ["stimulus_units",
                   "stimulus_duration",
                   "stimulus_amplitude",
                   "sweep_number",
                   "vm_delta_mv",
                   "pre_noise_rms_mv",
                   "slow_noise_rms_mv",
                   "stimulus_scale_factor",
                   "post_noise_rms_mv",
                   "slow_vm_mv",
                   "stimulus_code",
                   "stimulus_name",
                   ]


def generate_qc_input(se_input,se_output):

    qc_input = {}

    if 'stimulus_ontology_file' in se_input:
        qc_input['stimulus_ontology_file'] = se_input['stimulus_ontology_file']

    qc_input['qc_criteria'] = { "access_resistance_mohm_max":20.0,
                             "access_resistance_mohm_min":1.0,
                             "blowout_mv_max":10.0,
                             "blowout_mv_min":-10.0,
                             "created_at":"2015-01-29T13:51:29-08:00",
                             "electrode_0_pa_max":200.0,
                             "electrode_0_pa_min":-200.0,
                             "id":324256702,
                             "input_vs_access_resistance_max":0.15,
                             "leak_pa_max":100.0,
                             "leak_pa_min":-100.0,
                             "name":"Ephys QC Criteria v1.1",
                             "post_noise_rms_mv_max":0.07,
                             "pre_noise_rms_mv_max":0.07,
                             "seal_gohm_min":1.0,
                             "slow_noise_rms_mv_max":0.5,
                             "updated_at":"2015-01-29T13:51:29-08:00",
                             "vm_delta_mv_max":1.0}

    qc_input['sweep_features'] = sp.extract_sweep_features_subset(QC_INPUT_FEATURES,se_output['sweep_features'])

    qc_input['cell_features'] = se_output['cell_features']

    return qc_input


def main():
    """
    Usage:
    > python generate_qc_input.py --specimen_id SPECIMEN_ID --cell_dir CELL_DIR
    > python generate_qc_input.py --input_nwb_file input_nwb_file --cell_dir CELL_DIR

    """

    module = ags.ArgSchemaParser(schema_type=GeneratePipelineInputParameters)

    kwargs = module.args
    kwargs.pop("log_level")
    cell_dir = kwargs.pop("cell_dir")

    se_input = generate_se_input(**kwargs)

    ju.write(os.path.join(cell_dir,'se_input.json'), se_input)

    se_output = run_sweep_extraction(se_input["input_nwb_file"],
                                     se_input.get("input_h5_file",None),
                                     se_input.get("stimulus_ontology_file", None))

    ju.write(os.path.join(cell_dir,'se_output.json'),se_output)

    sp.drop_incomplete_sweeps(se_output["sweep_features"])
    sp.remove_sweep_feature("completed", se_output["sweep_features"])

    qc_input = generate_qc_input(se_input, se_output)

    ju.write(os.path.join(cell_dir,'qc_input.json'), qc_input)


if __name__=="__main__": main()
