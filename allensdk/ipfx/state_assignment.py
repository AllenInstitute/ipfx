import allensdk.internal.core.lims_utilities as lu

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
