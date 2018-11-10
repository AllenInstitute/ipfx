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
import pandas as pd
from . import ephys_extractor as efex
from . import ephys_data_set as eds
from . import spike_features as spkf
from . import stimulus_protocol_analysis as spa
from . import stim_features as stf
import error as er

DEFAULT_DETECTION_PARAMETERS = { 'dv_cutoff': 20.0, 'thresh_frac': 0.05 }

# DETECTION_PARAMETERS = {
#     eds.EphysDataSet.SHORT_SQUARE: { 'est_window': (1.02, 1.021), 'thresh_frac_floor': 0.1 },
#     eds.EphysDataSet.SHORT_SQUARE_TRIPLE: { 'est_window': (2.02, 2.021), 'thresh_frac_floor': 0.1 },
#     eds.EphysDataSet.RAMP: { 'start': 1.02 },
#     eds.EphysDataSet.LONG_SQUARE: { 'start': 1.02, 'end': 2.02 }
# }

DETECTION_PARAMETERS = {
    eds.EphysDataSet.SHORT_SQUARE: {'thresh_frac_floor': 0.1 },
    eds.EphysDataSet.RAMP: { },
    eds.EphysDataSet.LONG_SQUARE: { }
}





SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS = {
    20.0: -100.0,
    40.0: -200.0
    }

TEST_PULSE_DURATION_SEC = 0.4


def detection_parameters(stimulus_name):
    return DETECTION_PARAMETERS.get(stimulus_name, {})


def extractors_for_sweeps(sweep_set,
                          dv_cutoff=20., thresh_frac=0.05,
                          thresh_frac_floor=None,
                          est_window=None,
                          start=None, end=None):
    """Extract data from sweeps

    Parameters
    ----------
    sweep_set : SweepSet object

    dv_cutoff : float
    thresh_frac :
    thresh_frac_floor
    est_window :
    start :
    end :

    Returns
    -------
    spx : SpikeExtractor object
    spfx : SpikeTrainFeatureExtractor object

    """

    if est_window is not None:
        dv_cutoff, thresh_frac = spkf.estimate_adjusted_detection_parameters(sweep_set.v, sweep_set.t,
                                                                           est_window[0],
                                                                           est_window[1])

    if thresh_frac_floor is not None:
        thresh_frac = max(thresh_frac_floor, thresh_frac)

    spx = efex.SpikeExtractor(dv_cutoff=dv_cutoff, thresh_frac=thresh_frac, start=start, end=end)
    spfx = efex.SpikeTrainFeatureExtractor(start, end)

    return spx, spfx


def extract_sweep_features(data_set, sweep_table):
    sweep_groups = sweep_table.groupby(data_set.STIMULUS_NAME)[data_set.SWEEP_NUMBER]

    # extract sweep-level features
    sweep_features = {}

    for stimulus_name, sweep_numbers in sweep_groups:
        sweep_numbers = sorted(sweep_numbers)
        logging.debug("%s:%s" % (stimulus_name, ','.join(map(str, sweep_numbers))))

        sweep_set = data_set.sweep_set(sweep_numbers)

        dp = detection_parameters(stimulus_name).copy()
        for k in [ "start", "end" ]:
            if k in dp:
                dp.pop(k)

        spx, _ = extractors_for_sweeps(sweep_set, **dp)

        for sn, sweep in zip(sweep_numbers, sweep_set.sweeps):
#            logging.info("Extracting features from the sweep %d" % sn)
            spikes_df = spx.process(sweep.t, sweep.v, sweep.i)
            sweep_features[sn] = { 'spikes': spikes_df.to_dict(orient='records'), "id": sn }

    return sweep_features

def extract_cell_features(data_set,
                          ramp_sweep_numbers,
                          short_square_sweep_numbers,
                          long_square_sweep_numbers,
                          subthresh_min_amp):


    cell_features = {}

    # long squares
    if len(long_square_sweep_numbers) == 0:
        raise er.FeatureError("No long_square sweeps available for feature extraction")

    lsq_sweeps = data_set.sweep_set(long_square_sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(lsq_sweeps.sweeps[0].i, lsq_sweeps.sweeps[0].t)



    logging.info("Analyzing Long Square")
    logging.info("Long square stim start: %f, duration: %f", lsq_start, lsq_dur)

    lsq_spx, lsq_spfx = extractors_for_sweeps(lsq_sweeps,
                                              start = lsq_start,
                                              end = lsq_start+lsq_dur,
                                              **detection_parameters(data_set.LONG_SQUARE))
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=subthresh_min_amp)
    lsq_features = lsq_an.analyze(lsq_sweeps)
    cell_features["long_squares"] = lsq_an.as_dict(lsq_features, [ dict(id=sn) for sn in long_square_sweep_numbers ])

    if cell_features["long_squares"]["hero_sweep"] is None:
        raise er.FeatureError("Could not find hero sweep.")

    # short squares
    logging.info("Analyzing Short Square")
    if len(short_square_sweep_numbers) == 0:
        raise er.FeatureError("No short square sweeps available for feature extraction")

    ssq_sweeps = data_set.sweep_set(short_square_sweep_numbers)

    ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(ssq_sweeps.sweeps[0].i, ssq_sweeps.sweeps[0].t)
    logging.info("Short square stim start: %f, duration: %f", ssq_start, ssq_dur)
    ssq_spx, ssq_spfx = extractors_for_sweeps(ssq_sweeps,
                                              est_window = [ssq_start,ssq_start+0.001],
                                              **detection_parameters(data_set.SHORT_SQUARE))
    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
    ssq_features = ssq_an.analyze(ssq_sweeps)
    cell_features["short_squares"] = ssq_an.as_dict(ssq_features, [ dict(id=sn) for sn in short_square_sweep_numbers ])

    # ramps
    logging.info("Analyzing Ramps")
    if len(ramp_sweep_numbers) == 0:
        raise er.FeatureError("No ramp sweeps available for feature extraction")

    ramp_sweeps = data_set.sweep_set(ramp_sweep_numbers)

    ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
    logging.info("Ramp stim %f, %f", ramp_start, ramp_dur)

    ramp_spx, ramp_spfx = extractors_for_sweeps(ramp_sweeps,
                                                start = ramp_start,
                                                **detection_parameters(data_set.RAMP))
    ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
    ramp_features = ramp_an.analyze(ramp_sweeps)
    cell_features["ramps"] = ramp_an.as_dict(ramp_features, [ dict(id=sn) for sn in ramp_sweep_numbers ])

    return cell_features


def select_subthreshold_min_amplitude(stim_amps, decimals=0):
    """Find the min delta between amplitudes of coarse long square sweeps.  Includes failed sweeps.

    Parameters
    ----------
    stim_amps: list of stimulus amplitudes
    decimals: int of decimals to keep

    Returns
    -------
    subthresh_min_amp: float min amplitude
    min_amp_delta: min increment in the stimulus amplitude
    """

    amps_diff = np.round(np.diff(sorted(stim_amps)), decimals=decimals)

    amps_diff = amps_diff[amps_diff > 0]    # remove zeros, repeats are okay
    amp_deltas = np.unique(amps_diff)   # unique nonzero deltas

    if len(amp_deltas) == 0:
        raise IndexError("All stimuli have identical amplitudes")

    min_amp_delta = np.min(amp_deltas)

    if len(amp_deltas) != 1:
        logging.warning(
            "Found multiple coarse long square amplitude step differences: %s.  Using: %f" % (str(amp_deltas), min_amp_delta))

    subthresh_min_amp = SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS.get(min_amp_delta, None)

    if subthresh_min_amp is None:
        raise er.FeatureError("Unknown coarse long square amplitude delta: %f" % min_amp_delta)

    return subthresh_min_amp, min_amp_delta





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

#    print "slow trough t:", ephys_features["slow_trough_t_short_square"]

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

def categorize_iclamp_sweeps(data_set, stimuli_names):

    iclamp_st = data_set.filtered_sweep_table(current_clamp_only=True, stimuli=stimuli_names)

    mask = iclamp_st["passed"]==False
    failed_qc_st = iclamp_st[mask]
    failed_qc_sweeep_numbers = failed_qc_st["sweep_number"].sort_values().values

    mask = iclamp_st["passed"]==True
    passed_qc_st = iclamp_st[mask]
    passed_qc_sweeep_numbers = passed_qc_st["sweep_number"].sort_values().values

    logging.info("%s sweeps  passed: %s  failed: %s " % (stimuli_names[0],
                                                        str(passed_qc_sweeep_numbers),
                                                        str(failed_qc_sweeep_numbers)))

    if len(passed_qc_sweeep_numbers) == 0:
        raise er.FeatureError("No %s sweeps available for feature extraction " % stimuli_names[0])


    return passed_qc_sweeep_numbers




def extract_data_set_features(data_set, subthresh_min_amp=None):
    """

    Parameters
    ----------
    data_set : AibsDataSet
        data set
    subthresh_min_amp

    Returns
    -------
    cell_features :

    sweep_features :

    cell_record :

    sweep_records :

    """
    ontology = data_set.ontology
    # for logging purposes
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True)
    passed_iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True, passing_only=True)
    logging.info("%d of %d current-clamp sweeps passed QC", len(passed_iclamp_sweeps), len(iclamp_sweeps))

    if len(passed_iclamp_sweeps) == 0:
        raise er.FeatureError("There are no QC-passed sweeps available to analyze")

    # extract cell-level features
    logging.info("Computing cell features")

    clsq_sweeps = data_set.filtered_sweep_table(current_clamp_only=True, stimuli=ontology.coarse_long_square_names)
    clsq_sweep_numbers = clsq_sweeps['sweep_number'].sort_values().values

    lsq_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.long_square_names)
    ssq_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.short_square_names)
    ramp_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.ramp_names)

    if subthresh_min_amp is None:
        if len(clsq_sweep_numbers)>0:
            subthresh_min_amp, clsq_amp_delta = select_subthreshold_min_amplitude(clsq_sweeps['stimulus_amplitude'])
            logging.info("Coarse long squares: %f pA step size.  Using subthreshold minimum amplitude of %f.", clsq_amp_delta, subthresh_min_amp)
        else:
            subthresh_min_amp = -100
            logging.info("Assigned subthreshold minimum amplitude of %f.", subthresh_min_amp)


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

