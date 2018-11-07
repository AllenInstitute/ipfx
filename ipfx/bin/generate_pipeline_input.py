import json
import allensdk.internal.core.lims_utilities as lu
import argschema as ags
from ipfx._schemas import GeneratePipelineInputParameters
import ipfx.qc_protocol as qcp
import allensdk.core.json_utilities as ju
import os.path


def get_sweep_states_from_lims(specimen_id):

    res = lu.query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    sweep_states = []

    for sweep in res:
        # only care about manual calls
        if sweep["workflow_state"] == "manual_passed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': True})
        elif sweep["workflow_state"] == "manual_failed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': False})

    return sweep_states


def get_input_nwb_file_from_lims(specimen_id):

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


def get_input_h5_file_from_lims(specimen_id):

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


def make_input_json(args, plot_figures=False):

    pipe_input = dict()

    if "specimen_id" in args:
        pipe_input['input_nwb_file'] = get_input_nwb_file_from_lims(args["specimen_id"])
        pipe_input['input_h5_file'] = get_input_h5_file_from_lims(args["specimen_id"])
        pipe_input['manual_sweep_states'] = get_sweep_states_from_lims(args["specimen_id"])

    elif "input_nwb_file" in args:
        pipe_input['input_nwb_file'] = args["input_nwb_file"]
        pipe_input['manual_sweep_states'] = []

    cell_name = get_cell_name(pipe_input['input_nwb_file'])

    cell_dir = os.path.join(args["output_dir"],cell_name)
    if not os.path.exists(cell_dir):
        os.makedirs(cell_dir)

    if plot_figures:
        pipe_input['qc_fig_dir'] = os.path.join(cell_dir,"qc_figs")

    pipe_input['output_nwb_file'] = os.path.join(cell_dir, "output.nwb")
    pipe_input['qc_criteria'] = ju.read(qcp.DEFAULT_QC_CRITERIA_FILE)
    # pipe_args['stimulus_ontology_file'] = stimulus_ontology_file

    with open(os.path.join(cell_dir, 'pipeline_input.json'), 'w') as f:
        f.write(json.dumps(pipe_input, indent=2))


def get_cell_name(input_nwb_file,):

    input_nwb_file_basename = os.path.basename(input_nwb_file)
    cell_name = os.path.splitext(input_nwb_file_basename)[0]

    return cell_name


def main():


    module = ags.ArgSchemaParser(schema_type=GeneratePipelineInputParameters)

    make_input_json(module.args)

if __name__ == "__main__": main()



