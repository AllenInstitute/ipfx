import allensdk.core.json_utilities as ju
import argschema as ags
from ipfx._schemas import GeneratePipelineInputParameters
import os.path
from run_pipeline import run_pipeline
import generate_pipeline_input as gpi


INPUT_JSON = "pipeline_input.json"
OUTPUT_JSON = "pipeline_output.json"


def main():
    """
    Runs pipeline from the nwb file
    Usage:
    python pipeline_from_nwb_file.py --input_nwb_file INPUT_NWB_FILE --output_dir OUTPUT_DIR

    """

    module = ags.ArgSchemaParser(schema_type=GeneratePipelineInputParameters)

    gpi.make_input_json(module.args)
    cell_name = gpi.get_cell_name(module.args['input_nwb_file'])
    cell_dir = os.path.join(module.args["output_dir"],cell_name)
    input_json = os.path.join(cell_dir,INPUT_JSON)
    pipe_input = ju.read(input_json)

    output = run_pipeline(pipe_input["input_nwb_file"],
                          pipe_input.get("input_h5_file", None),
                          pipe_input["output_nwb_file"],
                          pipe_input.get("stimulus_ontology_file", None),
                          pipe_input.get("qc_fig_dir",None),
                          pipe_input["qc_criteria"],
                          pipe_input["manual_sweep_states"])

    output_json = os.path.join(cell_dir,OUTPUT_JSON)
    ju.write(output_json, output)

if __name__ == "__main__": main()



