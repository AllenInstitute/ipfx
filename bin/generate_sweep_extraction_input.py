import json
import allensdk.core.json_utilities as ju
import os
import sys
import allensdk.internal.core.lims_utilities as lu

specimen_id = int(sys.argv[1])

res = lu.query("""
select err.storage_directory||'EPHYS_FEATURE_EXTRACTION_V2_QUEUE_'||err.id||'_input.json' as input_json,
       err.storage_directory||err.id||'.nwb' as nwb_file,
       err.storage_directory||sp.name||'.h5' as h5_file
from specimens sp
join ephys_roi_results err on err.id = sp.ephys_roi_result_id
where sp.id = %d
""" % specimen_id)[0]
#
# /allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/EPHYS_FEATURE_EXTRACTION_V2_QUEUE_642966460_input.json
res = { k.decode('UTF-8'):v for k,v in res.items() }
print(res)
with open(res['input_json'], 'r') as f:
    d = json.load(f)



import allensdk.ipfx.ephys_data_set as eds
stimulus_ontology_file = eds.DEFAULT_STIMULUS_ONTOLOGY_FILE
test_dir = "specimen_%d" % specimen_id
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

d = {}
if os.path.exists(res['h5_file']):
    d['input_h5_file'] = res['h5_file']


test_data_dir = "./"
d['input_nwb_file'] = os.path.join(test_data_dir, res['nwb_file'])
d['stimulus_ontology_file'] = os.path.join(test_data_dir,stimulus_ontology_file)


with open(os.path.join(test_data_dir, 'sweep_extraction_input.json'), 'w') as f:
    f.write(json.dumps(d, indent=2))
