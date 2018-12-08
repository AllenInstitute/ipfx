import allensdk.core.json_utilities as ju
import os
from ipfx._schemas import GeneratePipelineInputParameters
import argschema as ags
import allensdk.internal.core.lims_utilities as lu
import make_stimulus_ontology as mso

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


def generate_se_input(cell_dir,
                      specimen_id=None,
                      input_nwb_file=None,
                      ):

    se_input = dict()

    if specimen_id:
        se_input['input_nwb_file'] = get_input_nwb_file_from_lims(specimen_id)
        se_input['input_h5_file'] = get_input_h5_file_from_lims(specimen_id)

    elif input_nwb_file:
        se_input['input_nwb_file'] = input_nwb_file

    stim_ontolgoy_tags = mso.make_stimulus_ontology_from_lims()
    stim_ontology_json = os.path.join(cell_dir, 'stimulus_ontology.json')
    ju.write(stim_ontology_json, stim_ontolgoy_tags)

    se_input["stimulus_ontology_file"] = stim_ontology_json

    return se_input


def main():

    """
    Usage:
    > python generate_se_input.py --specimen_id SPECIMEN_ID --cell_dir CELL_DIR
    > python generate_se_input.py --input_nwb_file input_nwb_file --cell_dir CELL_DIR

    """

    module = ags.ArgSchemaParser(schema_type=GeneratePipelineInputParameters)

    kwargs = module.args
    kwargs.pop("log_level")

    se_input = generate_se_input(**kwargs)

    input_json = os.path.join(kwargs["cell_dir"],'se_input.json')

    ju.write(input_json, se_input)


if __name__=="__main__": main()
