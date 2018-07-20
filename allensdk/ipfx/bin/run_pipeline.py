import os
import argschema as ags
from allensdk.ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

import allensdk.core.json_utilities as ju

import logging

def assign_sweep_states(manual_sweep_states, qc_sweep_states, out_sweep_data):
    sweep_states = { s["sweep_number"]:s["passed"] for s in qc_sweep_states }

    for mss in manual_sweep_states:
        sn = mss["sweep_number"]
        logging.debug("overriding sweep state for sweep number %d from %s to %s", sn, str(sweep_states[sn]), mss["passed"])
        sweep_states[sn] = mss["passed"]

    for sweep in out_sweep_data:
        sn = sweep["sweep_number"]
        if sn in sweep_states:
            sweep["passed"] = sweep_states[sn]
        else:
            logging.warning("could not find QC state for sweep number %d", sn)

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

    qc_output = run_qc(input_nwb_file,
                       input_h5_file,
                       stimulus_ontology_file,
                       se_output["cell_features"],
                       se_output["sweep_data"],
                       qc_criteria)

    logging.info("QC completed checks")
    assign_sweep_states(manual_sweep_states,
                        qc_output["sweep_states"],
                        se_output["sweep_data"])

    logging.info("Assigned sweep state")

    fx_output = run_feature_extraction(input_nwb_file,
                                       stimulus_ontology_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_data'],
                                       se_output['cell_features'])
    logging.info("Extracted features!")

    return dict( sweep_extraction=se_output,
                 qc=qc_output,
                 feature_extraction=fx_output )


def main():

    module = ags.ArgSchemaParser(schema_type=PipelineParameters)

    output = run_pipeline(module.args["input_nwb_file"],
                          module.args.get("input_h5_file", None),
                          module.args["output_nwb_file"],
                          module.args.get("stimulus_ontology_file", None),
                          module.args["qc_fig_dir"],
                          module.args["qc_criteria"],
                          module.args["manual_sweep_states"])


    ju.write(module.args["output_json"], output)



if __name__ == "__main__": main()
