import os
import allensdk.core.json_utilities as ju


"""
    Generates an input JSON for QC module saved in the MODULE_IO_DIR
"""

MODULE_IO_DIR = "../../tests/module_io/Ephys_Roi_Result_500844779"

FEATURE_SUBSET = ["stimulus_units",
               "stimulus_duration",
               "sweep_number",
               "vm_delta_mv",
               "pre_noise_rms_mv",
               "slow_noise_rms_mv",
               "stimulus_scale_factor",
               "post_noise_rms_mv",
               "slow_vm_mv",
               "stimulus_code",
               "stimulus_name",
               ]



def extract_sweep_features_subset(sweep_features, FEATURE_SUBSET):

    sweep_features_subset = []
    for sf in sweep_features:
        sf_subset = {k: sf[k] for k in FEATURE_SUBSET}
        sweep_features_subset.append(sf_subset)

    return sweep_features_subset


def main():

    se_output_json = os.path.join(MODULE_IO_DIR, 'se_output.json')
    se_input_json = os.path.join(MODULE_IO_DIR, 'se_input.json')

    outd = ju.read(se_output_json)
    inpd = ju.read(se_input_json)

    d = {}

    if 'stimulus_ontology_file' in inpd:
        d['stimulus_ontology_file'] = inpd['stimulus_ontology_file']

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

    d['sweep_features'] = extract_sweep_features_subset(outd['sweep_features'], FEATURE_SUBSET)
    d['cell_features'] = outd['cell_features']

    qc_input_json = os.path.join(MODULE_IO_DIR,'qc_input.json')

    ju.write(qc_input_json, d)


if __name__=="__main__": main()
