import pandas as pd
import numpy as np


def build_cell_feature_record(cell_features):

    cell_record = dict()

    cell_record["rheobase_sweep_num"] = cell_features["long_squares"]["rheobase_sweep"]["sweep_number"]
    cell_record["thumbnail_sweep_num"] = cell_features["long_squares"]["hero_sweep"]["sweep_number"]

    base = cell_features["long_squares"]
    cell_record["vrest"] = nan_get(base, "v_baseline")
    cell_record["ri"] = nan_get(base, "input_resistance")
    cell_record["sag"] = nan_get(base, "sag")
    cell_record["tau"] = nan_get(base, "tau")
    cell_record["vm_for_sag"] = nan_get(base, "vm_for_sag")
    cell_record["f_i_curve_slope"] = nan_get(base, "fi_fit_slope")

    # change the base to hero sweep
    base = cell_features["long_squares"]["hero_sweep"]
    cell_record["adaptation"] = nan_get(base, "adapt")
    cell_record["latency"] = nan_get(base, "latency")
    cell_record["avg_isi"] = nan_get(base, "mean_isi")

    spike_feature_names = ["upstroke_downstroke_ratio",
                             "peak_v", "peak_t",
                             "trough_v", "trough_t",
                             "fast_trough_v", "fast_trough_t",
                             "slow_trough_v", "slow_trough_t",
                             "threshold_v", "threshold_i", "threshold_t",
                             ]

    # mean feature of first spike for

    ls_rheo_spike_0 = cell_features["long_squares"]["rheobase_sweep"]["spikes"][0]
    add_features_to_record(spike_feature_names, ls_rheo_spike_0, cell_record, postfix="long_square")

    ramp_mean_spike_0 = cell_features["ramps"]["mean_spike_0"]
    add_features_to_record(spike_feature_names, ramp_mean_spike_0, cell_record, postfix="ramp")

    sq_mean_spike_0 = cell_features["short_squares"]["mean_spike_0"]
    add_features_to_record(spike_feature_names, sq_mean_spike_0, cell_record, postfix="short_square")

    cell_record["threshold_i_short_square"] = nan_get(cell_features["short_squares"], "stimulus_amplitude")

    convert_units(cell_record)

    return cell_record


def convert_units(cell_record):

    # convert to ms

    cell_record["avg_isi"] = (cell_record["avg_isi"] * 1e3) if cell_record["avg_isi"] is not None else None
    cell_record["tau"] = (cell_record["tau"] * 1e3) if cell_record["tau"] is not None else None


def add_features_to_record(feature_names, feature_data, cell_record, postfix=None):

    for feature_name in feature_names:
        feature_name_in_record = '{}_{}'.format(feature_name, postfix)
        cell_record[feature_name_in_record] = nan_get(feature_data,feature_name)


def build_sweep_feature_record(sweep_table, sweep_features):

    sweep_table = sweep_table.copy()

    num_spikes = []
    pds = []
    for sn in sweep_table['sweep_number']:
        sweep = sweep_features.get(sn, {})
        pds.append(sweep.get('peak_deflect', [None])[0])
        num_spikes.append(len(sweep.get('spikes',[])))

    sweep_table['peak_deflection'] = pd.Series(pds)
    sweep_table['num_spikes'] = pd.Series(num_spikes)

    return sweep_table.to_dict(orient='records')


def nan_get(obj, key):
    """ Return a value from a dictionary.  If it does not exist, return None.  If it is NaN, return None """
    v = obj.get(key, None)

    if v is None:
        return None
    else:
        return None if np.isnan(v) else v
