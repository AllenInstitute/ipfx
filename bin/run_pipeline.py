import os
import argschema as ags
from allensdk.ipfx._schemas import PipelineParameters
import allensdk.ipfx.state_assignment as stas

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

import allensdk.core.json_utilities as ju

import logging




def run_pipeline(specimen_id,
                 input_nwb_file,
                 input_h5_file,
                 output_nwb_file,
                 stimulus_ontology_file,
                 qc_fig_dir,
                 qc_criteria):

    se_output = run_sweep_extraction(input_nwb_file,
                                     input_h5_file,
                                     stimulus_ontology_file)
    print "sweeps extracted"

    qc_output = run_qc(input_nwb_file,
                       input_h5_file,
                       stimulus_ontology_file,
                       se_output["cell_features"],
                       se_output["sweep_data"],
                       qc_criteria)

    se_output = stas.assign_state(se_output,
                             qc_output)

    sweep_states_lims = stas.get_sweep_state_from_lims(specimen_id)
    se_output = stas.overwrite_sweep_state_from_lims(se_output, sweep_states_lims)

    fx_output = run_feature_extraction(input_nwb_file,
                                       stimulus_ontology_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_data'],
                                       se_output['cell_features'])

    return dict( sweep_extraction=se_output,
                 qc=qc_output,
                 feature_extraction=fx_output )


def main():

    module = ags.ArgSchemaParser(schema_type=PipelineParameters)
    print "ontology:", module.args["stimulus_ontology_file"]
    logging.info("specimen_id: %d", module.args["specimen_id"])



    output = run_pipeline(module.args["specimen_id"],
                          module.args["input_nwb_file"],
                          module.args.get("input_h5_file", None),
                          module.args["output_nwb_file"],
                          module.args["stimulus_ontology_file"],
                          module.args["qc_fig_dir"],
                          module.args["qc_criteria"])

    ju.write(module.args["output_json"], output)



if __name__ == "__main__": main()
