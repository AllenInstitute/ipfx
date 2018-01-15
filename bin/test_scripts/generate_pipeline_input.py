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
with open(res['input_json'], 'r') as f:
    d = json.load(f)

stim_names = {}
for stim_type in d[1]:
    stim_name = stim_type['name']
    for code in stim_type['ephys_raw_stimulus_names']:
        stim_names[code['name']] = stim_name

stimulus_ontology_file = "stimulus_ontology.json" 
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
d['qc_criteria'] = { "access_resistance_mohm_max":20.0, 
                     "access_resistance_mohm_min":1.0,
                     "blowout_mv_max":10.0,
                     "blowout_mv_min":-10.0,
                     "created_at":"2015-01-29T13:51:29-08:00",
                     "electrode_0_pa_max":200.0,
                     "electrode_0_pa_min":-200.0,
                     "id":324256702,
                     "input_vs_access_resistance_max":0.15,
                     "leak_pa_max":100.0,
                     "leak_pa_min":-100.0,
                     "name":"Ephys QC Criteria v1.1",
                     "post_noise_rms_mv_max":0.07,
                     "pre_noise_rms_mv_max":0.07,
                     "seal_gohm_min":1.0,
                     "slow_noise_rms_mv_max":0.5,
                     "updated_at":"2015-01-29T13:51:29-08:00",
                     "vm_delta_mv_max":1.0}

with open(os.path.join(test_dir, 'pipeline_input.json'), 'w') as f:
    f.write(json.dumps(d, indent=2))
