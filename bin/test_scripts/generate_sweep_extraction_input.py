import json
import allensdk.core.json_utilities as ju

with open('/allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/EPHYS_FEATURE_EXTRACTION_V2_QUEUE_642966460_input.json', 'r') as f:
    d = json.load(f)

stim_names = {}
for stim_type in d[1]:
    stim_name = stim_type['name']
    for code in stim_type['ephys_raw_stimulus_names']:
        stim_names[code['name']] = stim_name

ontology_file = "test_scripts/stimulus_ontology.json"

ju.write(ontology_file, stim_names)

data = { 'input_nwb_file': "/allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/642966460.nwb",
         'stimulus_ontology_file': ontology_file
         }

ju.write('test/sweep_extraction_input.json', data)
