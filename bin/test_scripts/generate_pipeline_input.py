import json
import os
import allensdk.internal.core.lims_utilities as lu
import sys

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

stim_names = {}
for stim_type in d[1]:
    stim_name = stim_type['name']
    for code in stim_type['ephys_raw_stimulus_names']:
        stim_names[code['name']] = stim_name

import allensdk.ipfx.ephys_data_set as eds
stimulus_ontology_file = eds.DEFAULT_STIMULUS_ONTOLOGY_FILE
test_dir = "specimen_%d" % specimen_id
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

d = {}
if os.path.exists(res['h5_file']):
    d['input_h5_file'] = res['h5_file']

d['input_nwb_file'] = res['nwb_file']
d['output_nwb_file'] = os.path.join(test_dir, "output.nwb")
d['qc_fig_dir'] = os.path.join(test_dir,"qc_figs")
d['stimulus_ontology_file'] = stimulus_ontology_file
import allensdk.ipfx.qc_features as qcf
import allensdk.core.json_utilities as ju
d['qc_criteria'] = ju.read(qcf.DEFAULT_QC_CRITERIA_FILE)

with open(os.path.join(test_dir, 'pipeline_input.json'), 'w') as f:
    f.write(json.dumps(d, indent=2))
