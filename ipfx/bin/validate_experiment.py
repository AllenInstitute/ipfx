import allensdk.core.json_utilities as ju
import numpy as np
import sys
import os
import logging


def nullisclose(a, b):
    if a is None or b is None:
        return a is None and b is None
    else:
        return np.isclose(a, b)

def validate_feature_set(features, d1, d2):

    for f in features:
        v1 = d1[f]
        v2 = d2[f]

        if isinstance(v1, unicode):
            if v1 != v2:
                print((f, v1, v2))
        elif not nullisclose(v1, v2):
            print((f, v1, v2))


def get_pipeline_output_json(storage_dir,err_id):
    """Return the name of the output file giving preference to the newer version
    Parameters
    ----------
    storage_dir: str of storage directory
    err_id: str err_id
    Returns
    -------
    pipeline_output_json: str filename
    """

    pipeline_output_v2_json = os.path.join(storage_dir, "EPHYS_FEATURE_EXTRACTION_V2_QUEUE_%s_output.json" % err_id)
    pipeline_output_v1_json = os.path.join(storage_dir, "EPHYS_FEATURE_EXTRACTION_QUEUE_%s_output.json" % err_id)

    # if the input_v2_json does not exist, then use input_v1_json instead:
    if os.path.isfile(pipeline_output_v2_json):
        pipeline_output_json = pipeline_output_v2_json
    else:
        pipeline_output_json = pipeline_output_v1_json


    return pipeline_output_json


def validate_run_completion(pipeline_input_json, pipeline_output_json):
    """Check if the pipeline output was generated as way of confirming that the run completed
    Parameters
    ----------
    pipeline_input_json
    pipeline_output_json
    Returns
    -------
    """

    pipeline_input = ju.read(pipeline_input_json)
    if os.path.isfile(pipeline_output_json):
        logging.info("run completed")
    else:
        logging.info("run failed")


def validate_pipeline(input_json, output_json):
    input_data = ju.read(input_json)


    storage_dir = os.path.dirname(input_data["input_nwb_file"])
    err_id,_ = os.path.splitext(os.path.basename(input_data["input_nwb_file"]))

    pipeline_output_json = get_pipeline_output_json(storage_dir,err_id)

    print("pipeline output json:", pipeline_output_json)

    pipeline_output = ju.read(pipeline_output_json)
    test_output = ju.read(output_json)

    validate_cell_features(pipeline_output,
                           pipeline_output["specimens"][0]["ephys_features"][0],
                           test_output["feature_extraction"]["cell_record"])


def validate_se(test_output_json="test/sweep_extraction_output.json"):
    print("**** SWEEP EXTRACTION")
    pipeline_output_json = "/allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/EPHYS_NWB_STIMULUS_SUMMARY_QUEUE_642966460_output.json"

    pipeline_output = ju.read(pipeline_output_json)
    test_output = ju.read(test_output_json)

    sweep_features = [ "stimulus_interval", "post_vm_mv", "pre_vm_mv", "stimulus_duration", "stimulus_start_time", "sweep_number", "vm_delta_mv", "leak_pa", "pre_noise_rms_mv", "slow_noise_rms_mv", "post_noise_rms_mv", "slow_vm_mv", "stimulus_amplitude", "stimulus_units", "bridge_balance_mohm" ]

    test_sweeps = { s['sweep_number']:s for s in test_output['sweep_data'] }
    for d1 in pipeline_output['sweep_data'].values():
        try:
            d2 = test_sweeps[d1['sweep_number']]
            validate_feature_set(sweep_features, d1, d2)
        except KeyError as e:
            print(e)

    other_sweep_features = [ "stimulus_name", "clamp_mode", "stimulus_scale_factor", "stimulus_code" ]


def validate_cell_features(err, ephys_features, cell_record):
    po_features = [ 'input_resistance_mohm', 'input_access_resistance_ratio', 'seal_gohm', 'initial_access_resistance_mohm', 'blowout_mv', 'electrode_0_pa' ]

    validate_feature_set(po_features, err, cell_record)

    ef_features = [ 'tau', 'threshold_t_long_square', 'thumbnail_sweep_id', 'threshold_v_ramp', 'peak_v_short_square', 'avg_isi', 'sag', 'slow_trough_v_ramp', 'adaptation', 'trough_t_ramp', 'trough_v_long_square', 'thumbnail_sweep_num', 'rheobase_sweep_id', 'latency', 'rheobase_sweep_num', 'fast_trough_v_ramp', 'trough_t_long_square', 'slow_trough_v_long_square', 'threshold_t_short_square', 'peak_t_ramp', 'upstroke_downstroke_ratio_short_square', 'threshold_v_long_square', 'fast_trough_t_long_square', 'ri', 'threshold_v_short_square', 'upstroke_downstroke_ratio_ramp', 'vm_for_sag', 'threshold_i_long_square', 'peak_t_long_square', 'threshold_i_short_square', 'slow_trough_t_long_square', 'peak_v_ramp', 'fast_trough_t_short_square', 'fast_trough_t_ramp', 'threshold_i_ramp', 'slow_trough_v_short_square', 'peak_t_short_square', 'slow_trough_t_short_square', 'trough_v_short_square', 'slow_trough_t_ramp', 'f_i_curve_slope', 'trough_t_short_square', 'threshold_t_ramp', 'fast_trough_v_long_square', 'upstroke_downstroke_ratio_long_square', 'trough_v_ramp', 'peak_v_long_square', 'fast_trough_v_short_square', 'vrest']

    validate_feature_set(ef_features, ephys_features, cell_record)


def validate_fx(test_output_json="test/fx_output.json"):
    print("**** FX")
    pipeline_output_json = "/allen/programs/celltypes/production/humancelltypes/prod242/Ephys_Roi_Result_642966460/EPHYS_FEATURE_EXTRACTION_V2_QUEUE_642966460_output.json"

    pipeline_output = ju.read(pipeline_output_json)
    test_output = ju.read(test_output_json)

    validate_cell_features(pipeline_output,
                           pipeline_output['specimens'][0]['ephys_features'][0],
                           test_output['cell_record'])


def main():
    print("Validating experiment...")
    pij, poj = sys.argv[1:3]
    #validate_se()
    #validate_fx()
    validate_run_completion(pij,poj)
    validate_pipeline(pij, poj)

if __name__ == "__main__": main()

