import os
import argschema as ags
from allensdk.ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction
from run_visualization import run_visualization

import allensdk.core.json_utilities as ju
import logging


def assign_sweep_states(manual_sweep_states, qc_sweep_states, out_sweep_props):
    sweep_states = { s["sweep_number"]:s["passed"] for s in qc_sweep_states }

    for mss in manual_sweep_states:
        sn = mss["sweep_number"]
        logging.debug("overriding sweep state for sweep number %d from %s to %s", sn, str(sweep_states[sn]), mss["passed"])
        sweep_states[sn] = mss["passed"]

    for sweep in out_sweep_props:
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
                       se_output["sweep_features"],
                       qc_criteria)
    logging.info("QC checks completed")

    assign_sweep_states(manual_sweep_states,
                        qc_output["sweep_states"],
                        se_output["sweep_features"]
                        )
    logging.info("Assigned sweep state")

    fx_output = run_feature_extraction(input_nwb_file,
                                       stimulus_ontology_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_features'],
                                       se_output['cell_features'])
    logging.info("Extracted features!")

    # run_visualization(input_nwb_file,
    #                   stimulus_ontology_file,
    #                   qc_fig_dir,
    #                   se_output["sweep_features"],
    #                   fx_output)
    # logging.info("Visualized results!")

    se_output['sweep_data'] = se_output.pop('sweep_features') # for backward compatibility only
    # On Windows int64 keys of sweep numbers cannot be converted to str by json.dump when serializing. Thus, we are converting here:
    fx_output["sweep_features"] = {str(k): v for k, v in fx_output["sweep_features"].items()}

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
