import logging
import argschema as ags
from ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

import allensdk.core.json_utilities as ju
import ipfx.sweep_props as sp


def run_pipeline(input_nwb_file,
                 input_h5_file,
                 output_nwb_file,
                 stimulus_ontology_file,
                 qc_fig_dir,
                 qc_criteria,
                 manual_sweep_states):

    se_output = run_sweep_extraction(input_nwb_file,
                                     input_h5_file,
                                     stimulus_ontology_file)
    logging.info("Computed QC features")

    sp.drop_incomplete_sweeps(se_output["sweep_features"])
    sp.remove_sweep_feature("completed", se_output["sweep_features"])

    qc_output = run_qc(stimulus_ontology_file,
                       se_output["cell_features"],
                       se_output["sweep_features"],
                       qc_criteria)
    logging.info("QC checks completed")

    sp.override_auto_sweep_states(manual_sweep_states,qc_output["sweep_states"])
    sp.assign_sweep_states(qc_output["sweep_states"],se_output["sweep_features"])

    logging.info("Assigned sweep states")

    fx_output = run_feature_extraction(input_nwb_file,
                                       stimulus_ontology_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_features'],
                                       se_output['cell_features'],
                                       )
    logging.info("Extracted features!")

    se_output['sweep_data'] = se_output.pop('sweep_features') # for backward compatibility only

    # On Windows int64 keys of sweep numbers cannot be converted to str by json.dump when serializing.
    # Thus, we are converting them here:
    fx_output["sweep_features"] = {str(k): v for k, v in fx_output["sweep_features"].items()}

    return dict( sweep_extraction=se_output,
                 qc=qc_output,
                 feature_extraction=fx_output )


def main():

    """
    Usage:
    python run_pipeline_extraction.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=PipelineParameters)

    output = run_pipeline(module.args["input_nwb_file"],
                          module.args.get("input_h5_file", None),
                          module.args["output_nwb_file"],
                          module.args.get("stimulus_ontology_file", None),
                          module.args.get("qc_fig_dir", None),
                          module.args["qc_criteria"],
                          module.args["manual_sweep_states"])


    ju.write(module.args["output_json"], output)



if __name__ == "__main__": main()
