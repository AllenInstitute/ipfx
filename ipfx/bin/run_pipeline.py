import logging
import argschema as ags
from ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

import allensdk.core.json_utilities as ju
import ipfx.sweep_props as sp
import ipfx.logging_utils as lu


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

    sp.drop_tagged_sweeps(se_output["sweep_features"])
    sp.remove_sweep_feature("tags",se_output["sweep_features"])

    qc_output = run_qc(stimulus_ontology_file,
                       se_output["cell_features"],
                       se_output["sweep_features"],
                       qc_criteria)

    if qc_output["cell_state"]["failed_qc"]:
        logging.warning("Failed QC. No ephys features extracted.")

        return dict(sweep_extraction=se_output,
                    qc=qc_output,
                    )

    sp.override_auto_sweep_states(manual_sweep_states, qc_output["sweep_states"])
    sp.assign_sweep_states(qc_output["sweep_states"], se_output["sweep_features"])

    fx_output = run_feature_extraction(input_nwb_file,
                                       stimulus_ontology_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_features'],
                                       se_output['cell_features'],
                                       )

    return dict(sweep_extraction=se_output,
                qc=qc_output,
                feature_extraction=fx_output
                )


def main():

    """
    Usage:
    python run_pipeline.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=PipelineParameters)

    output = run_pipeline(module.args["input_nwb_file"],
                          module.args.get("input_h5_file", None),
                          module.args["output_nwb_file"],
                          module.args.get("stimulus_ontology_file", None),
                          module.args.get("qc_fig_dir", None),
                          module.args.get("qc_criteria", None),
                          module.args.get("manual_sweep_states", None))


    ju.write(module.args["output_json"], output)

    lu.log_pretty_header("Analysis completed!", level=1)


if __name__ == "__main__": main()
