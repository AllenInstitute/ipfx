# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import logging
import six
import pandas as pd
from . import ephys_extractor as efex
from . import ephys_features as ft
from . import ephys_data_set as eds

HERO_MIN_AMP_OFFSET = 39.0
HERO_MAX_AMP_OFFSET = 61.0

DEFAULT_DETECTION_PARAMETERS = { 'dv_cutoff': 20.0, 'thresh_frac': 0.05 }

DETECTION_PARAMETERS = {
    eds.EphysDataSet.SHORT_SQUARE: { 'est_window': (1.02, 1.021), 'thresh_frac_floor': 0.1 },
    eds.EphysDataSet.SHORT_SQUARE_TRIPLE: { 'est_window': (2.02, 2.021), 'thresh_frac_floor': 0.1 },
    eds.EphysDataSet.RAMP: { 'fixed_start': 1.02 },
    eds.EphysDataSet.LONG_SQUARE: { 'fixed_start': 1.02, 'fixed_end': 2.02 }
}   

MEAN_FEATURES = [ "upstroke_downstroke_ratio", "peak_v", "peak_t", "trough_v", "trough_t",
                  "fast_trough_v", "fast_trough_t", "slow_trough_v", "slow_trough_t",
                  "threshold_v", "threshold_i", "threshold_t", "peak_v", "peak_t" ]

SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS = { 
    20.0: -100.0,
    40.0: -200.0 
    }

TEST_PULSE_DURATION_SEC = 0.4

def extractor_for_data_set_sweeps(data_set, sweep_numbers,
                                  fixed_start=None, fixed_end=None,                             
                                  dv_cutoff=20., thresh_frac=0.05, 
                                  thresh_frac_floor=None,
                                  est_window=None):

    v_set = []
    t_set = []
    i_set = []

    start = []
    end = []
    
    for sweep_number in sweep_numbers:
        data = data_set.sweep(sweep_number)
        v = data['response'] * 1e3 # mV
        i = data['stimulus'] * 1e12 # pA
        hz = data['sampling_rate']
        dt = 1. / hz
        t = np.arange(0, len(v)) * dt # sec

        s, e = dt * np.array(data['index_range'])
        v_set.append(v)
        i_set.append(i)
        t_set.append(t)
        start.append(s)
        end.append(e)

    if est_window is not None:
        dv_cutoff, thresh_frac = ft.estimate_adjusted_detection_parameters(v_set, t_set, 
                                                                           est_window[0], 
                                                                           est_window[1])

    if thresh_frac_floor is not None:
        thresh_frac = max(thresh_frac_floor, thresh_frac)

    if fixed_start and not fixed_end:
        start = [fixed_start] * len(end)
    elif fixed_start and fixed_end:
        start = fixed_start
        end = fixed_end

    return efex.EphysSweepSetFeatureExtractor(t_set, v_set, i_set, start=start, end=end,
                                              dv_cutoff=dv_cutoff, thresh_frac=thresh_frac,
                                              id_set=sweep_numbers)
  

def extract_sweep_features(data_set, sweep_table):
    sweep_groups = sweep_table.groupby(data_set.STIMULUS_NAME)[data_set.SWEEP_NUMBER]

    # extract sweep-level features
    sweep_features = {}

    for stimulus_name, sweep_numbers in sweep_groups:
        sweep_numbers = sorted(sweep_numbers)
        logging.debug("%s:%s" % (stimulus_name, ','.join(map(str, sweep_numbers))))

        detection_params = DETECTION_PARAMETERS.get(stimulus_name, {})

        # we want to detect spikes everywhere
        for k in ('fixed_start', 'fixed_end'):
            if k in detection_params:
                detection_params.pop(k)

        fex = extractor_for_data_set_sweeps(data_set, sweep_numbers, **detection_params)
        fex.process_spikes()

        sweep_features.update({ f.id:f.as_dict() for f in fex.sweeps() })

    return sweep_features

def find_hero_sweep(rheo_amp, long_square_spiking_sweeps, 
                    min_offset=HERO_MIN_AMP_OFFSET, 
                    max_offset=HERO_MAX_AMP_OFFSET):
    hero_min, hero_max = rheo_amp + min_offset, rheo_amp + max_offset
    hero_amp = float("inf")
    hero_sweep = None

    for sweep in long_square_spiking_sweeps:
        nspikes = len(sweep.spikes())
        amp = sweep.sweep_feature("stim_amp")

        if nspikes > 0 and amp > hero_min and amp < hero_max and amp < hero_amp:
            hero_amp = amp
            hero_sweep = sweep

    if hero_sweep is None:
        raise ft.FeatureError("Could not find hero sweep.")

    return hero_sweep
    
def extract_cell_features(data_set,
                          ramp_sweep_numbers,
                          short_square_sweep_numbers,
                          long_square_sweep_numbers,
                          subthresh_min_amp):


    cell_features = {}

    # long squares
    sweep = data_set.sweep(long_square_sweep_numbers[0])
    lsq_start, lsq_dur, _, _, _ = get_stim_characteristics(sweep['stimulus'], np.arange(0,len(sweep['stimulus']))/sweep['sampling_rate'])
    logging.info("Long square stim %f, %f", lsq_start, lsq_dur)

    if len(long_square_sweep_numbers) == 0:
        raise ft.FeatureError("no long_square sweep numbers provided")

    ext = extractor_for_data_set_sweeps(data_set, long_square_sweep_numbers, fixed_start=lsq_start, fixed_end=lsq_start+lsq_dur)
    long_squares_ext = efex.LongSquareFeatureExtractor(ext, subthresh_min_amp=subthresh_min_amp)
    long_squares_ext.analyze()
    cell_features["long_squares"] = long_squares_ext.as_dict()

    # short squares
    sweep = data_set.sweep(short_square_sweep_numbers[0])
    ssq_start, ssq_dur, _, _, _ = get_stim_characteristics(sweep['stimulus'], np.arange(0,len(sweep['stimulus']))/sweep['sampling_rate'])
    logging.info("Short square stim %f, %f", ssq_start, ssq_dur)

    if len(short_square_sweep_numbers) == 0:
        raise ft.FeatureError("no short square sweep numbers provided")

    ext = extractor_for_data_set_sweeps(data_set, short_square_sweep_numbers, **DETECTION_PARAMETERS[data_set.SHORT_SQUARE])
    short_squares_ext = efex.ShortSquareFeatureExtractor(ext)
    short_squares_ext.analyze()
    cell_features["short_squares"] = short_squares_ext.as_dict()

    # ramps
    sweep = data_set.sweep(ramp_sweep_numbers[0])
    ramp_start, ramp_dur, _, _, _ = get_stim_characteristics(sweep['stimulus'], np.arange(0,len(sweep['stimulus']))/sweep['sampling_rate'])
    logging.info("Ramp stim %f, %f", ramp_start, ramp_dur)

    if len(ramp_sweep_numbers) == 0:
        raise ft.FeatureError("no ramp sweep numbers provided")

    ext = extractor_for_data_set_sweeps(data_set, ramp_sweep_numbers, **DETECTION_PARAMETERS[data_set.RAMP])
    ramps_ext = efex.RampFeatureExtractor(ext)
    ramps_ext.analyze()
    cell_features["ramps"] = ramps_ext.as_dict()

    # find hero sweep
    hero_sweep = find_hero_sweep(cell_features['long_squares']['rheobase_i'], 
                                 long_squares_ext.suprathreshold_extractor.sweeps())
    adapt = hero_sweep.sweep_feature("adapt")
    latency = hero_sweep.sweep_feature("latency")
    mean_isi = hero_sweep.sweep_feature("mean_isi")

    # find the mean features of the first spike for the ramps and short squares
    ramps_ms0 = mean_features_spike_zero(ramps_ext.extractor.sweeps())
    ss_ms0 = mean_features_spike_zero(short_squares_ext.extractor.sweeps())

    # compute baseline from all long square sweeps
    v_baseline = np.mean(long_squares_ext.extractor.sweep_features('v_baseline'))

    cell_features['long_squares']['v_baseline'] = v_baseline
    cell_features['long_squares']['hero_sweep'] = hero_sweep.as_dict() if hero_sweep else None
    cell_features["ramps"]["mean_spike_0"] = ramps_ms0
    cell_features["short_squares"]["mean_spike_0"] = ss_ms0

    return cell_features

def mean_features_spike_zero(sweeps):
    """ Compute mean feature values for the first spike in list of extractors """

    output = {}
    for mf in MEAN_FEATURES:
        mfd = [ sweep.spikes()[0][mf] for sweep in sweeps if sweep.sweep_feature("avg_rate") > 0 ]
        output[mf] = np.mean(mfd)
    return output

def get_stim_characteristics(i, t, no_test_pulse=False):
    '''
    Identify the start time, duration, amplitude, start index, and
    end index of a general stimulus.
    This assumes that there is a test pulse followed by the stimulus square.
    '''

    di = np.diff(i)
    diff_idx = np.flatnonzero(di)# != 0)

    if len(diff_idx) == 0:
        return (None, None, 0.0, None, None)

    # skip the first up/down
    idx = 0 if no_test_pulse else 2

    # shift by one to compensate for diff()
    start_idx = diff_idx[idx] + 1
    end_idx = diff_idx[-1] + 1
    
    stim_start = float(t[start_idx])
    stim_dur = float(t[end_idx] - t[start_idx])
    stim_amp = float(i[start_idx])

    return (stim_start, stim_dur, stim_amp, start_idx, end_idx)



def select_subthreshold_min_amplitude(stim_amps, decimals=0):
    amp_diff = np.round(np.diff(sorted(stim_amps)), decimals=decimals)

    if len(np.unique(amp_diff)) != 1:
        raise ft.FeatureError("Long square sweeps must have even amplitude step differences")
    
    amp_delta = amp_diff[0]

    subthresh_min_amp = SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS.get(amp_delta, None)

    if subthresh_min_amp is None:
        raise ft.FeatureError("Unknown coarse long square amplitude delta: %f" % long_square_amp_delta)

    return subthresh_min_amp, amp_delta

def build_cell_feature_record(cell_features, sweep_features):
    ephys_features = {}

    # find hero and rheo sweeps in sweep table
    rheo_sweep_num = cell_features["long_squares"]["rheobase_sweep"]["id"]
    rheo_sweep_id = sweep_features.get(rheo_sweep_num, {}).get('id', None)

    if rheo_sweep_id is None:
        raise Exception("Could not find id of rheobase sweep number %d." % rheo_sweep_num)

    hero_sweep = cell_features["long_squares"]["hero_sweep"]
    if hero_sweep is None:
        raise Exception("Could not find hero sweep")
    
    hero_sweep_num = hero_sweep["id"]
    hero_sweep_id = sweep_features.get(hero_sweep_num, {}).get('id', None)

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

def build_sweep_feature_records(sweep_table, sweep_features):
    
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

    
def extract_data_set_features(data_set):
    # extract sweep-level features
    logging.debug("Computing sweep features")

    # for logging purposes
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True)
    passed_iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True, passing_only=True)
    logging.info("%d of %d current-clamp sweeps passed QC", len(passed_iclamp_sweeps), len(iclamp_sweeps))

    if len(passed_iclamp_sweeps) == 0:
        raise ft.FeatureError("There are no QC-passed sweeps available to analyze")

    # extract cell-level features
    logging.info("Computing cell features")
    lsq_sweeps = data_set.filtered_sweep_table(passing_only=True, current_clamp_only=True, stimulus_names=data_set.long_square_names)
    ssq_sweeps = data_set.filtered_sweep_table(passing_only=True, current_clamp_only=True, stimulus_names=data_set.short_square_names)
    ramp_sweeps = data_set.filtered_sweep_table(passing_only=True, current_clamp_only=True, stimulus_names=data_set.ramp_names)
    clsq_sweeps = data_set.filtered_sweep_table(current_clamp_only=True, stimulus_codes=data_set.coarse_long_square_codes)

    lsq_sweep_numbers = lsq_sweeps['sweep_number'].sort_values().values
    clsq_sweep_numbers = clsq_sweeps['sweep_number'].sort_values().values
    ssq_sweep_numbers = ssq_sweeps['sweep_number'].sort_values().values
    ramp_sweep_numbers = ramp_sweeps['sweep_number'].sort_values().values

    logging.debug("long square sweeps: %s", str(lsq_sweep_numbers))
    logging.debug("coarse long square sweeps: %s", str(clsq_sweep_numbers))
    logging.debug("short square sweeps: %s", str(ssq_sweep_numbers))
    logging.debug("ramp sweeps: %s", str(ramp_sweep_numbers))

    subthresh_min_amp, clsq_amp_delta = select_subthreshold_min_amplitude(clsq_sweeps['stimulus_amplitude'])
    logging.info("Long squares using %fpA step size.  Using subthreshold minimum amplitude of %f.", clsq_amp_delta, subthresh_min_amp)

    cell_features = extract_cell_features(data_set, 
                                          ramp_sweep_numbers,
                                          ssq_sweep_numbers,
                                          lsq_sweep_numbers,
                                          subthresh_min_amp)

    # compute sweep features
    logging.info("Computing sweep features")
    sweep_features = extract_sweep_features(data_set, iclamp_sweeps)

    # shuffle peak deflection for the subthreshold long squares
    for s in cell_features["long_squares"]["subthreshold_sweeps"]:
        sweep_features[s['id']]['peak_deflect'] = s['peak_deflect']

    cell_record = build_cell_feature_record(cell_features, sweep_features)
    sweep_records = build_sweep_feature_records(data_set.sweep_table, sweep_features)

    return cell_features, sweep_features, cell_record, sweep_records
