#!/usr/bin/python
import sys, logging
import os
import json
import shutil
import copy
import numpy as np
import shutil
import pandas as pd

import argschema as ags

import aibs.ipfx.experiment_features as efx
from aibs.ipfx.ephys_data_set import EphysDataSet
from aibs.ipfx._schemas import FeatureExtractionParameters

import allensdk.core.nwb_data_set 

def nan_get(obj, key):
    """ Return a value from a dictionary.  If it does not exist, return None.  If it is NaN, return None """
    v = obj.get(key, None)

    if v is None:
        return None
    else:
        return None if np.isnan(v) else v

def update_output_sweep_features(cell_features, sweep_features, sweep_index):
    # add peak deflection for subthreshold long squares
    for sweep_number, sweep in sweep_index.iteritems():
        pd = sweep_features.get(sweep_number,{}).get('peak_deflect', None)
        if pd is not None:
            sweep['peak_deflection'] = pd[0]
        
    # update num_spikes
    for sweep_num in sweep_features:
        num_spikes = len(sweep_features[sweep_num]['spikes'])
        if num_spikes == 0:
            num_spikes = None
        sweep_index[sweep_num]['num_spikes'] = num_spikes


def write_json(file_name, obj):
    # write output json file
    with open(file_name, 'w') as f:
        s = json.dumps(output_data,
                       indent=2,
                       ignore_nan=True,
                       default=json_handler,
                       iterable_as_array=True)
        f.write(s)

    
def json_handler(obj):
    """ Used by write_json convert a few non-standard types to things that the json package can handle. """
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif (isinstance(obj, np.bool) or
          isinstance(obj, np.bool_)):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        raise TypeError(
            'Object of type %s with value of %s is not JSON serializable' %
            (type(obj), repr(obj)))

def generate_output_cell_features(cell_features, sweep_features, sweep_index):
    ephys_features = {}

    # find hero and rheo sweeps in sweep table
    rheo_sweep_num = cell_features["long_squares"]["rheobase_sweep"]["id"]
    rheo_sweep_id = sweep_index.get(rheo_sweep_num, {}).get('id', None)

    if rheo_sweep_id is None:
        raise Exception("Could not find id of rheobase sweep number %d." % rheo_sweep_num)

    hero_sweep = cell_features["long_squares"]["hero_sweep"]
    if hero_sweep is None:
        raise Exception("Could not find hero sweep")
    
    hero_sweep_num = hero_sweep["id"]
    hero_sweep_id = sweep_index.get(hero_sweep_num, {}).get('id', None)

    if hero_sweep_id is None:
        raise Exception("Could not find id of hero sweep number %d." % hero_sweep_num)
    
    # create a table of values 
    # this is a dictionary of ephys_features
    base = cell_features["long_squares"]
    ephys_features["rheobase_sweep_id"] = rheo_sweep_id
    ephys_features["rheobase_sweep_num"] = rheo_sweep_num
    ephys_features["thumbnail_sweep_id"] = hero_sweep_id
    ephys_features["thumbnail_sweep_num"] = hero_sweep_num
    ephys_features["vrest"] = nan_get(base, "v_baseline")
    ephys_features["ri"] = nan_get(base, "input_resistance")

    # change the base to hero sweep
    base = cell_features["long_squares"]["hero_sweep"]
    ephys_features["adaptation"] = nan_get(base, "adapt")
    ephys_features["latency"] = nan_get(base, "latency")

    # convert to ms
    mean_isi = nan_get(base, "mean_isi")
    ephys_features["avg_isi"] = (mean_isi * 1e3) if mean_isi is not None else None

    # now grab the rheo spike
    base = cell_features["long_squares"]["rheobase_sweep"]["spikes"][0]
    ephys_features["upstroke_downstroke_ratio_long_square"] = nan_get(base, "upstroke_downstroke_ratio") 
    ephys_features["peak_v_long_square"] = nan_get(base, "peak_v")
    ephys_features["peak_t_long_square"] = nan_get(base, "peak_t")
    ephys_features["trough_v_long_square"] = nan_get(base, "trough_v")
    ephys_features["trough_t_long_square"] = nan_get(base, "trough_t")
    ephys_features["fast_trough_v_long_square"] = nan_get(base, "fast_trough_v")
    ephys_features["fast_trough_t_long_square"] = nan_get(base, "fast_trough_t")
    ephys_features["slow_trough_v_long_square"] = nan_get(base, "slow_trough_v")
    ephys_features["slow_trough_t_long_square"] = nan_get(base, "slow_trough_t")
    ephys_features["threshold_v_long_square"] = nan_get(base, "threshold_v")
    ephys_features["threshold_i_long_square"] = nan_get(base, "threshold_i")
    ephys_features["threshold_t_long_square"] = nan_get(base, "threshold_t")
    ephys_features["peak_v_long_square"] = nan_get(base, "peak_v")
    ephys_features["peak_t_long_square"] = nan_get(base, "peak_t")

    base = cell_features["long_squares"]
    ephys_features["sag"] = nan_get(base, "sag")
    # convert to ms
    tau = nan_get(base, "tau")
    ephys_features["tau"] = (tau * 1e3) if tau is not None else None 
    ephys_features["vm_for_sag"] = nan_get(base, "vm_for_sag")
    ephys_features["has_burst"] = None#base.get("has_burst", None)
    ephys_features["has_pause"] = None#base.get("has_pause", None)
    ephys_features["has_delay"] = None#base.get("has_delay", None)
    ephys_features["f_i_curve_slope"] = nan_get(base, "fi_fit_slope")

    # change the base to ramp
    base = cell_features["ramps"]["mean_spike_0"] # mean feature of first spike for all of these
    ephys_features["upstroke_downstroke_ratio_ramp"] = nan_get(base, "upstroke_downstroke_ratio")
    ephys_features["peak_v_ramp"] = nan_get(base, "peak_v")
    ephys_features["peak_t_ramp"] = nan_get(base, "peak_t")
    ephys_features["trough_v_ramp"] = nan_get(base, "trough_v")
    ephys_features["trough_t_ramp"] = nan_get(base, "trough_t")
    ephys_features["fast_trough_v_ramp"] = nan_get(base, "fast_trough_v")
    ephys_features["fast_trough_t_ramp"] = nan_get(base, "fast_trough_t")
    ephys_features["slow_trough_v_ramp"] = nan_get(base, "slow_trough_v")
    ephys_features["slow_trough_t_ramp"] = nan_get(base, "slow_trough_t")

    ephys_features["threshold_v_ramp"] = nan_get(base, "threshold_v")
    ephys_features["threshold_i_ramp"] = nan_get(base, "threshold_i")
    ephys_features["threshold_t_ramp"] = nan_get(base, "threshold_t")

    # change the base to short_square
    base = cell_features["short_squares"]["mean_spike_0"] # mean feature of first spike for all of these
    ephys_features["upstroke_downstroke_ratio_short_square"] = nan_get(base, "upstroke_downstroke_ratio")
    ephys_features["peak_v_short_square"] = nan_get(base, "peak_v")
    ephys_features["peak_t_short_square"] = nan_get(base, "peak_t")

    ephys_features["trough_v_short_square"] = nan_get(base, "trough_v")
    ephys_features["trough_t_short_square"] = nan_get(base, "trough_t")

    ephys_features["fast_trough_v_short_square"] = nan_get(base, "fast_trough_v")
    ephys_features["fast_trough_t_short_square"] = nan_get(base, "fast_trough_t")

    ephys_features["slow_trough_v_short_square"] = nan_get(base, "slow_trough_v")
    ephys_features["slow_trough_t_short_square"] = nan_get(base, "slow_trough_t")

    ephys_features["threshold_v_short_square"] = nan_get(base, "threshold_v")
    #ephys_features["threshold_i_short_square"] = nan_get(base, "threshold_i")
    ephys_features["threshold_t_short_square"] = nan_get(base, "threshold_t")

    ephys_features["threshold_i_short_square"] = nan_get(cell_features["short_squares"], "stimulus_amplitude")

    return ephys_features

def embed_spike_times(input_nwb_file, output_nwb_file, sweep_list, sweep_features):
    # embed spike times in NWB file
    logging.debug("Embedding spike times")
    tmp_nwb_file = output_nwb + ".tmp"

    shutil.copy(input_nwb_file, tmp_nwb_file)
    for sweep in sweep_list:
        sweep_num = sweep['sweep_number']

        if sweep_num not in sweep_features:
            continue

        try:
            spikes = sweep_features[sweep_num]['spikes']
            spike_times = [ s['threshold_t'] for s in spikes ]
            NwbDataSet(tmp_nwb_file).set_spike_times(sweep_num, spike_times)
        except Exception as e:
            logging.info("sweep %d has no sweep features. %s", sweep_num, e.message)

    try:
        shutil.move(tmp_nwb_file, out_nwb_file)
    except OSError as e:
        logging.error("Problem renaming file: %s -> %s" % (tmp_nwb_file, out_nwb_file))
        raise e

class PipelineDataSet(EphysDataSet):
    def __init__(self, sweep_list, file_name):
        self.sweep_list = sweep_list
        self.data_set = allensdk.core.nwb_data_set.NwbDataSet(file_name)

    def get_sweep_table(self):
        return pd.DataFrame(self.sweep_list)

    def get_sweep(self, sweep_number):
        return self.data_set.get_sweep(sweep_number)
            

def main():
    module = ags.ArgSchemaParser(schema_type=FeatureExtractionParameters)
    args = module.args

    input_nwb_file = args["input_nwb_file"]
    output_nwb_file = args["output_nwb_file"]
    qc_fig_dir = args["qc_fig_dir"]
    sweep_list = args["sweep_list"]
    
    data_set = PipelineDataSet(sweep_list, input_nwb_file)

    cell_features, sweep_features = efx.extract_experiment_features(data_set)
    embed_spike_times(input_nwb_file, output_nwb_file, data_set.sweep_info(iclamp_only=True), sweep_features)
    efx.save_qc_figures(qc_fig_dir, nwb_file, output_data, True)
    update_output_sweep_features(cell_features, sweep_features, sweep_index)
    ephys_features = generate_output_cell_features(cell_features, sweep_features, sweep_index)

    write_json(args["output_json"], output_data)

            
if __name__ == "__main__": main()
