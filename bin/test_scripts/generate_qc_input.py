import json
with open('test/sweep_extraction_output.json', 'r') as f:
    outd = json.load(f)

with open('test/sweep_extraction_input.json', 'r') as f:
    ind = json.load(f)

d = {}
d['input_nwb_file'] = ind['input_nwb_file']
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

d['sweep_data'] = outd['sweep_data']
d['cell_features'] = outd['cell_features']

with open('test/qc_input.json', 'w') as f:
    f.write(json.dumps(d, indent=2))
