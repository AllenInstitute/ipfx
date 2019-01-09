import os
import warnings

try:
    import allensdk.internal.core.lims_utilities as lu
except ImportError as e:
    warnings.warn("Can not import LIMS libraries due to '%s', using default values." % e)
    lu = None

def get_input_nwb_file(specimen_id):

    if not lu:
        return None

    query="""
    select err.storage_directory||'EPHYS_FEATURE_EXTRACTION_V2_QUEUE_'||err.id||'_input.json' as input_v2_json,
           err.storage_directory||'EPHYS_FEATURE_EXTRACTION_QUEUE_'||err.id||'_input.json' as input_v1_json,
           err.storage_directory||err.id||'.nwb' as nwb_file
    from specimens sp
    join ephys_roi_results err on err.id = sp.ephys_roi_result_id
    where sp.id = %d
    """ % specimen_id
    res = lu.query(query)[0]
    res = { k.decode('UTF-8'):v for k,v in res.items() }

    # if the input_v2_json does not exist, then use input_v1_json instead:
    if os.path.isfile(res["input_v2_json"]):
        res["input_json"] = res["input_v2_json"]
    else:
        res["input_json"] = res["input_v1_json"]

    nwb_file_name  = res['nwb_file']

    return nwb_file_name


def get_input_h5_file(specimen_id):

    if not lu:
        return None

    h5_res = lu.query("""
    select err.*, wkf.*,sp.name as specimen_name 
    from ephys_roi_results err 
    join specimens sp on sp.ephys_roi_result_id = err.id 
    join well_known_files wkf on wkf.attachable_id = err.id 
    where sp.id = %d 
    and wkf.well_known_file_type_id = 306905526
    """ % specimen_id)

    h5_file_name = os.path.join(h5_res[0]['storage_directory'], h5_res[0]['filename']) if len(h5_res) else None

    return h5_file_name


def get_sweep_states(specimen_id):

    sweep_states = []

    if not lu:
        return sweep_states

    res = lu.query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    for sweep in res:
        # only care about manual calls
        if sweep["workflow_state"] == "manual_passed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': True})
        elif sweep["workflow_state"] == "manual_failed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': False})

    return sweep_states


def get_stimuli_description():

    if not lu:
        return None

    stims = lu.query("""
    select ersn.name as stimulus_code, est.name as stimulus_name from ephys_raw_stimulus_names ersn
    join ephys_stimulus_types est on ersn.ephys_stimulus_type_id = est.id
    """)

    return stims

