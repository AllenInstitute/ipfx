import os
import argschema as ags
from allensdk.ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

import allensdk.core.json_utilities as ju
import allensdk.internal.core.lims_utilities as lu

import logging


def get_sweep_state_from_lims(specimen_id):

    res = lu.query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    sweep_state = {}

    for sweep in res:

        sweep_number = sweep["sweep_number"]
        sweep_state[sweep_number] = {"workflow_state": sweep["workflow_state"]}

        if sweep["workflow_state"] in ['manual_passed', 'auto_passed']:

            sweep_state[sweep_number] = {"passed": True, "workflow_state": sweep["workflow_state"]}
        else:
            sweep_state[sweep_number] = {"passed": False, "workflow_state": sweep["workflow_state"]}


    return sweep_state


def overwrite_sweep_state_from_lims(se_output, sweep_states):
    """

    Parameters
    ----------
    se_output: dict of sweep extraction properties
    sweep_states: dict of sweep states

    Returns
    -------
    se_output: dict of sweep extraction properties

    """
    for sweep in se_output["sweep_data"]:
        sweep['passed'] = sweep_states[sweep['sweep_number']]['passed']

    return se_output



def assign_state(se_output,qc_output):

    """Assign state to sweeps and a cell from qc

    Parameters
    ----------
    se_output: dict of sweep properties
    qc_output: dict of qc properties

    Returns
    -------
        se_output: dict of sweep properties
    """
    sweep_states = { s['sweep_number']:s for s in qc_output['sweep_states'] }

    for sweep in se_output["sweep_data"]:
        sweep['passed'] = sweep_states[sweep['sweep_number']]['passed']

    se_output['cell_features']['cell_state'] = qc_output['cell_state']

    return se_output


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

    qc_output = run_qc(input_nwb_file,
                       input_h5_file,
                       stimulus_ontology_file,
                       se_output["cell_features"],
                       se_output["sweep_data"],
                       qc_criteria)

    se_output = assign_state(se_output,
                             qc_output)

    sweep_states_lims = get_sweep_state_from_lims(specimen_id)
    se_output = overwrite_sweep_state_from_lims(se_output, sweep_states_lims)

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
