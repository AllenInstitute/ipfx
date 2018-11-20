import os
import allensdk.core.json_utilities as ju
from run_pipeline import assign_sweep_states
from ipfx._schemas import GeneratePipelineInputParameters
import argschema as ags
from run_sweep_extraction import run_sweep_extraction
from generate_qc_input import generate_qc_input
from generate_se_input import generate_se_input
from run_qc import run_qc
import generate_pipeline_input as gpi


FX_INPUT_FEATURE = [
                    "stimulus_code",
                    "stimulus_name",
                    "stimulus_amplitude",
                    "sweep_number",
                    "stimulus_units",
                    "bridge_balance_mohm",
                    "leak_pa",
                    "passed",
                    "truncated",
                    "pre_vm_mv"
                    ]


def generate_fx_input(se_input,
                      se_output,
                      args,
                      plot_figures=False):

    fx_input = dict()

    fx_input['input_nwb_file'] = se_input['input_nwb_file']

    if 'input_h5_file' in se_input:
        fx_input['input_h5_file'] = se_input['input_h5_file']

    fx_input['sweep_features'] = gpi.extract_sweep_features_subset(se_output["sweep_features"], FX_INPUT_FEATURE)

    fx_input['cell_features'] = se_output['cell_features']

    if plot_figures:
        fx_input['qc_fig_dir'] = os.path.join(args["cell_dir"], "qc_figs")

    fx_input['output_nwb_file'] = os.path.join(args["cell_dir"], "output.nwb")

    return fx_input


def main():
    """
    Usage:
    > python generate_fx_input.py --specimen_id SPECIMEN_ID --cell_dir CELL_DIR
    > python generate_fx_input.py --input_nwb_file INPUT_NWB_FILE --cell_dir CELL_DIR

    """

    module = ags.ArgSchemaParser(schema_type=GeneratePipelineInputParameters)

    se_input = generate_se_input(module.args)
    ju.write(os.path.join(module.args["cell_dir"],'se_input.json'), se_input)

    se_output = run_sweep_extraction(se_input["input_nwb_file"],
                                     se_input.get("input_h5_file",None),
                                     se_input.get("stimulus_ontology_file", None))

    ju.write(os.path.join(module.args["cell_dir"],'se_output.json'),se_output)

    qc_input = generate_qc_input(se_input, se_output)
    ju.write(os.path.join(module.args["cell_dir"],'qc_input.json'), qc_input)

    qc_output = run_qc(qc_input.get("stimulus_ontology_file",None),
                       qc_input["cell_features"],
                       qc_input["sweep_features"],
                       qc_input["qc_criteria"])
    ju.write(os.path.join(module.args["cell_dir"],'qc_output.json'), qc_output)

    manual_sweep_states = []
    assign_sweep_states(manual_sweep_states,
                        qc_output["sweep_states"],
                        se_output["sweep_features"]
                        )

    fx_input = generate_fx_input(se_input,
                                 se_output,
                                 module.args,
                                 plot_figures=True)

    ju.write(os.path.join(module.args["cell_dir"],'fx_input.json'), fx_input)

if __name__=="__main__": main()






