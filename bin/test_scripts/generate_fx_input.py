import json, os
import allensdk.core.json_utilities as ju

#pipeline_json = "/allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/EPHYS_FEATURE_EXTRACTION_V2_QUEUE_642966460_input.json"
extract_input_json = "test/sweep_extraction_input.json"
extract_output_json = "test/sweep_extraction_output.json"
qc_output_json = "test/qc_output.json"

input_json = "test/fx_input.json"

extract_input = ju.read(extract_input_json)
extract_output = ju.read(extract_output_json)
qc_output = ju.read(qc_output_json)

sweeps = extract_output['sweep_data']
sweep_states = { s['sweep_number']:s for s in qc_output['sweep_states'] }

for sweep in sweeps:
    sweep['passed'] = sweep_states[sweep['sweep_number']]['passed']

outdata= { 
    'input_nwb_file': extract_input['input_nwb_file'],
    'output_nwb_file': 'test/output.nwb',
    'qc_fig_dir': 'test/qc_figs',
    'sweep_list': sweeps,
    'cell_features': extract_output['cell_features']
    }

ju.write(input_json, outdata)



    



