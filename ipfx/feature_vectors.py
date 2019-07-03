from __future__ import absolute_import
import numpy as np
import logging
import pandas as pd
from scipy import stats
from . import stim_features as stf
from . import data_set_features as dsf
from . import stimulus_protocol_analysis as spa
from . import time_series_utils as tsu
from . import error as er


def extract_feature_vectors(data_set,
                            ramp_sweep_numbers,
                            short_square_sweep_numbers,
                            long_square_sweep_numbers,
                            use_lsq=True, use_ssq=True, use_ramp=True,
                            target_sampling_rate=50000, ap_window_length=0.003):
    """Extract feature vectors for downstream dimensionality reduction

    Parameters
    ----------
    data_set : AibsDataSet
    ramp_sweep_numbers :
    short_square_sweep_numbers :
    long_square_sweep_numbers :

    Returns
    -------
    all_features : dictionary of feature vectors
    """

    # long squares
    if use_lsq:
        if len(long_square_sweep_numbers) == 0:
            raise er.FeatureError("No long_square sweeps available for feature extraction")

        check_lsq_sweeps = data_set.sweep_set(long_square_sweep_numbers)
        lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(check_lsq_sweeps.sweeps[0].i, check_lsq_sweeps.sweeps[0].t)

        # Check that all sweeps are long enough and not ended early
        extra_dur = 0.2
        good_lsq_sweep_numbers = [n for n, s in zip(long_square_sweep_numbers, check_lsq_sweeps.sweeps)
                                  if s.t[-1] >= lsq_start + lsq_dur + extra_dur and not np.all(s.v[tsu.find_time_index(s.t, lsq_start + lsq_dur)-100:tsu.find_time_index(s.t, lsq_start + lsq_dur)] == 0)]
        lsq_sweeps = data_set.sweep_set(good_lsq_sweep_numbers)

        lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                      start = lsq_start,
                                                      end = lsq_start + lsq_dur,
                                                      **dsf.detection_parameters(data_set.LONG_SQUARE))
        lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)
        lsq_features = lsq_an.analyze(lsq_sweeps)
    else:
        lsq_sweeps = None
        lsq_features = None

    # short squares
    if use_ssq:
        if len(short_square_sweep_numbers) == 0:
            raise er.FeatureError("No short square sweeps available for feature extraction")

        ssq_sweeps = data_set.sweep_set(short_square_sweep_numbers)

        ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(ssq_sweeps.sweeps[0].i, ssq_sweeps.sweeps[0].t)
        ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                                      est_window = [ssq_start, ssq_start+0.001],
                                                      **dsf.detection_parameters(data_set.SHORT_SQUARE))
        ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
        ssq_features = ssq_an.analyze(ssq_sweeps)
    else:
        ssq_sweeps = None
        ssq_features = None

    # ramps
    if use_ramp:
        if len(ramp_sweep_numbers) == 0:
            raise er.FeatureError("No ramp sweeps available for feature extraction")

        ramp_sweeps = data_set.sweep_set(ramp_sweep_numbers)

        ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
        ramp_spx, ramp_spfx = dsf.extractors_for_sweeps(ramp_sweeps,
                                                    start = ramp_start,
                                                    **dsf.detection_parameters(data_set.RAMP))
        ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
        ramp_features = ramp_an.analyze(ramp_sweeps)
    else:
        ramp_sweeps = None
        ramp_features = None

    all_features = feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                                   lsq_features, ssq_features, ramp_features,
                                   lsq_start, lsq_start + lsq_dur,
                                   lsq_spx,
                                   target_sampling_rate=target_sampling_rate,
                                   ap_window_length=ap_window_length)
    return all_features


def extract_multipatch_feature_vectors(lsq_supra_sweeps, lsq_supra_start, lsq_supra_end,
                                       lsq_sub_sweeps, lsq_sub_start, lsq_sub_end,
                                       target_sampling_rate=50000, ap_window_length=0.003):
    lsq_supra_spx, lsq_supra_spfx = dsf.extractors_for_sweeps(lsq_supra_sweeps, start=lsq_supra_start, end=lsq_supra_end)
    lsq_supra_an = spa.LongSquareAnalysis(lsq_supra_spx, lsq_supra_spfx, subthresh_min_amp=-100., require_subthreshold=False)
    lsq_supra_features = lsq_supra_an.analyze(lsq_supra_sweeps)

    lsq_sub_spx, lsq_sub_spfx = dsf.extractors_for_sweeps(lsq_sub_sweeps, start=lsq_sub_start, end=lsq_sub_end)
    lsq_sub_an = spa.LongSquareAnalysis(lsq_sub_spx, lsq_sub_spfx, subthresh_min_amp=-100., require_suprathreshold=False)
    lsq_sub_features = lsq_sub_an.analyze(lsq_sub_sweeps)

    all_features = feature_vectors_multipatch(
        lsq_supra_sweeps, lsq_supra_features,
        lsq_supra_start, lsq_supra_end,
        lsq_sub_sweeps, lsq_sub_features,
        lsq_sub_start, lsq_sub_end,
        lsq_supra_spx,
        target_sampling_rate=target_sampling_rate,
        ap_window_length=ap_window_length
    )
    return all_features


def feature_vectors_multipatch(lsq_supra_sweeps, lsq_supra_features,
                               lsq_supra_start, lsq_supra_end,
                               lsq_sub_sweeps, lsq_sub_features,
                               lsq_sub_start, lsq_sub_end,
                               lsq_spike_extractor,
                               amp_tolerance=4.,
                               feature_width=20, rate_width=50,
                               target_sampling_rate=50000, ap_window_length=0.003):
    """Feature vectors from stimulus set features"""

    result = {}
    result["step_subthresh"] = step_subthreshold(lsq_sub_sweeps, lsq_sub_features, lsq_sub_start, lsq_sub_end, amp_tolerance=amp_tolerance)
    result["subthresh_norm"] = subthresh_norm(lsq_sub_sweeps, lsq_sub_features, lsq_sub_start, lsq_sub_end)
    result["subthresh_depol_norm"] = subthresh_depol_norm(lsq_supra_sweeps, lsq_supra_features, lsq_supra_start, lsq_supra_end)
    result["isi_shape"] = isi_shape(lsq_supra_sweeps, lsq_supra_features, duration=lsq_supra_end - lsq_supra_start)
    result["first_ap"] = first_ap_features(lsq_supra_sweeps, None, None,
                                           lsq_supra_features, None, None,
                                           target_sampling_rate=target_sampling_rate,
                                           window_length=ap_window_length)
    result["spiking"] = spiking_features(lsq_supra_sweeps, lsq_supra_features, lsq_spike_extractor,
                                         0., 1.,
                                         feature_width, rate_width, amp_tolerance=amp_tolerance)
    return result


def feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                    lsq_features, ssq_features, ramp_features,
                    lsq_start, lsq_end, lsq_spike_extractor,
                    feature_width=20, rate_width=50,
                    target_sampling_rate=50000, ap_window_length=0.003):
    """Feature vectors from stimulus set features"""

    result = {}
    result["step_subthresh"] = step_subthreshold(lsq_sweeps, lsq_features, lsq_start, lsq_end, amp_tolerance=5)
    result["subthresh_norm"] = subthresh_norm(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    result["subthresh_depol_norm"] = subthresh_depol_norm(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    result["isi_shape"] = isi_shape(lsq_sweeps, lsq_features)
    result["first_ap"] = first_ap_features(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                                           lsq_features, ssq_features, ramp_features,
                                           target_sampling_rate=target_sampling_rate,
                                           window_length=ap_window_length)
    result["spiking"] = spiking_features(lsq_sweeps, lsq_features, lsq_spike_extractor,
                                         lsq_start, lsq_end,
                                         feature_width, rate_width)
    return result


def identify_subthreshold_hyperpol_with_amplitudes(features, sweeps):
    """ Identify subthreshold responses from hyperpolarizing steps

        Parameters
        ----------
        features : output of LongSquareAnalysis.analyze()
        sweeps: SweepSet

        Returns:
        amp_sweep_dict : dict of sweeps with amplitudes as keys
        deflect_dict : dict of (base, deflect) tuples with amplitudes as keys
    """

    # Get non-spiking sweeps
    subthresh_df = features["subthreshold_sweeps"]

    # Get responses to hyperpolarizing steps
    subthresh_df = subthresh_df.loc[subthresh_df["stim_amp"] < 0] # only consider hyperpolarizing steps

    # Construct dictionary
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweeps.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)
    subthresh_deflect = subthresh_df["peak_deflect"].values
    subthresh_base = subthresh_df["v_baseline"].values

    mask = subthresh_amps < -1000 # TEMP QC ISSUE: Check for bad amps; shouldn't have to do this in general
    subthresh_amps = subthresh_amps[~mask]
    subthresh_sweeps = subthresh_sweeps[~mask]
    subthresh_deflect = subthresh_deflect[~mask]
    subthresh_base = subthresh_base[~mask]

    amp_sweep_dict = dict(zip(subthresh_amps, subthresh_sweeps))
    base_deflect_tuples = zip(subthresh_base, [d[0] for d in subthresh_deflect])
    deflect_dict = dict(zip(subthresh_amps, base_deflect_tuples))
    return amp_sweep_dict, deflect_dict


def step_subthreshold(amp_sweep_dict, target_amps, start, end,
                      extend_duration=0.2, subsample_interval=0.01,
                      amp_tolerance=0.):
    """ Subsample set of subthreshold step responses including regions before and after step

        Parameters
        ----------
        amp_sweep_dict : dict of amplitude / sweep pairs

        target_amps: list of desired amplitudes for output vector
        features : output of LongSquareAnalysis.analyze()
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        extend_duration : in seconds (default 0.2)
        subsample_interval : in seconds (default 0.01)

        Returns
        -------
        output_vector : subsampled, concatenated voltage trace
    """

    # Subsample each sweep
    subsampled_dict = {}
    for amp in amp_sweep_dict:
        swp = amp_sweep_dict[amp]
        start_index = tsu.find_time_index(swp.t, start - extend_duration)
        delta_t = swp.t[1] - swp.t[0]
        subsample_width = int(np.round(subsample_interval / delta_t))
        end_index = tsu.find_time_index(swp.t, end + extend_duration)
        subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
        subsampled_dict[amp] = subsampled_v

    extend_length = int(np.round(extend_duration / subsample_interval))
    available_amps = np.array(list(subsampled_dict.keys()))
    output_list = []
    for amp in target_amps:
        amp_diffs = np.array(np.abs(available_amps - amp))

        if np.any(amp_diffs <= amp_tolerance):
            matching_amp = available_amps[np.argmin(amp_diffs)]
            logging.debug("found amp of {} to match {} (tol={})".format(matching_amp, amp, amp_tolerance))
            output_list.append(subsampled_dict[matching_amp])
        else:
            lower_amp = 0
            upper_amp = 0
            for a in available_amps:
                if a < amp:
                    if lower_amp == 0:
                        lower_amp = a
                    elif a > lower_amp:
                        lower_amp = a
                if a > amp:
                    if upper_amp == 0:
                        upper_amp = a
                    elif a < upper_amp:
                        upper_amp = a
            if lower_amp != 0 and upper_amp != 0:
                logging.debug("interpolating for amp {} with lower {} and upper {}".format(amp, lower_amp, upper_amp))
                avg = (subsampled_dict[lower_amp] + subsampled_dict[upper_amp]) / 2.
                scale = amp / ((lower_amp + upper_amp) / 2.)
                base_v = avg[:extend_length].mean()
                avg[extend_length:-extend_length] = (avg[extend_length:-extend_length] - base_v) * scale + base_v
            elif lower_amp != 0:
                logging.debug("interpolating for amp {} from lower {}".format(amp, lower_amp))
                avg = subsampled_dict[lower_amp].copy()
                scale = amp / lower_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            elif upper_amp != 0:
                logging.debug("interpolating for amp {} from upper {}".format(amp, upper_amp))
                avg = subsampled_dict[upper_amp].copy()
                scale = amp / upper_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            output_list.append(avg)

    return np.hstack(output_list)


def subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg


def subthresh_norm(amp_sweep_dict, deflect_dict, start, end, target_amp=-101.,
                   extend_duration=0.2, subsample_interval=0.01):
    """ Subthreshold step response closest to target amplitude normalized to baseline and peak deflection

        Parameters
        ----------
        amp_sweep_dict : dict of amplitude / sweep pairs
        deflect_dict : dict of (baseline, deflect) tuples with amplitude keys
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        target_amp: search target for amplitude, pA (default -101)
        extend_duration: in seconds (default 0.2)
        subsample_interval: in seconds (default 0.01)

        Returns
        -------
        subsampled_v : subsampled, normalized voltage trace
    """
    available_amps = np.array(list(amp_sweep_dict.keys()))

    sweep_ind = np.argmin(np.abs(available_amps - target_amp))
    matching_amp = available_amps[sweep_ind]
    swp = amp_sweep_dict[matching_amp]
    base, deflect_v = deflect_dict[matching_amp]
    delta = base - deflect_v

    start_index = tsu.find_time_index(swp.t, start - extend_duration)
    delta_t = swp.t[1] - swp.t[0]
    subsample_width = int(np.round(subsample_interval / delta_t))
    end_index = tsu.find_time_index(swp.t, end + extend_duration)
    subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v


def subthresh_depol_norm(sweep_set, features, start, end,
    extend_duration=0.2, subsample_interval=0.01, steady_state_interval=0.1):
    """ Largest positive-going subthreshold step response that does not evoke spikes,
        normalized to baseline and steady-state at end of step

        Parameters
        ----------
        sweep_set : SweepSet
        features : output of LongSquareAnalysis.analyze()
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        extend_duration: in seconds (default 0.2)
        subsample_interval: in seconds (default 0.01)

        Returns
        -------
        subsampled_v : subsampled, normalized voltage trace
    """
    if "subthreshold_sweeps" in features.keys():
        sweep_table = features["subthreshold_sweeps"]
    else:
        all_sweeps_table = features["sweeps"]
        sweep_table = all_sweeps_table.loc[all_sweeps_table["avg_rate"] == 0, :]

    amps = np.rint(sweep_table["stim_amp"].values)
    if np.sum(amps > 0) == 0:
        logging.debug("No subthreshold depolarizing sweeps found - returning all-zeros response")
        logging.debug(features["sweeps"][["stim_amp", "avg_rate"]])

        # create all-zeros response of appropriate length
        swp = sweep_set.sweeps[0]
        delta_t = swp.t[1] - swp.t[0]
        start_index = tsu.find_time_index(swp.t, start - extend_duration)
        delta_t = swp.t[1] - swp.t[0]
        subsample_width = int(np.round(subsample_interval / delta_t))
        end_index = tsu.find_time_index(swp.t, end + extend_duration)
        subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
        subsampled_v = np.zeros_like(subsampled_v)
        return subsampled_v

    subthresh_sweep_ind = sweep_table.index.tolist()
    subthresh_sweeps = np.array(sweep_set.sweeps)[subthresh_sweep_ind]
    max_sweep_ind = np.argmax(amps)
    swp = subthresh_sweeps[max_sweep_ind]

    base = sweep_table.at[sweep_table.index[max_sweep_ind], "v_baseline"]

    interval_start_index = tsu.find_time_index(swp.t, end - steady_state_interval)
    interval_end_index = tsu.find_time_index(swp.t, end)
    steady_state_v = swp.v[interval_start_index:interval_end_index].mean()

    delta = steady_state_v - base

    start_index = tsu.find_time_index(swp.t, start - extend_duration)
    delta_t = swp.t[1] - swp.t[0]
    subsample_width = int(np.round(subsample_interval / delta_t))
    end_index = tsu.find_time_index(swp.t, end + extend_duration)
    subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v


def isi_shape(sweep_set, features, duration=1., n_points=100, min_spike=5):
    """ Average interspike voltage trajectory with normalized duration, aligned to threshold

        Parameters
        ----------
        sweep_set : SweepSet
        features : output of LongSquareAnalysis.analyze()
        n_points: number of points in output
        min_spike: minimum number of spikes for first preference sweep

        Returns
        -------
        isi_norm : averaged, threshold-aligned, and duration-normalized voltage trace
    """
    sweep_table = features["sweeps"]
    mask_supra = sweep_table["stim_amp"].values >= features["rheobase_i"]
    amps = np.rint(sweep_table.loc[mask_supra, "stim_amp"].values - features["rheobase_i"])

    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike (use 100 ms after spike or end of trace)

    only_one_spike = False
    n_spikes = sweep_table.loc[mask_supra, "avg_rate"].values * duration

    mask_spikes = n_spikes >= min_spike
    if np.any(mask_spikes):
        min_amp_masked = np.argmin(amps[mask_spikes])
        isi_index = np.arange(0, len(n_spikes), dtype=int)[mask_spikes][min_amp_masked]
    elif np.any(n_spikes > 1):
        isi_index = np.argmax(n_spikes)
    else:
        only_one_spike = True
        isi_index = np.argmin(amps)

    isi_sweep = np.array(sweep_set.sweeps)[mask_supra][isi_index]
    spikes_set_index = np.arange(len(sweep_set.sweeps))[mask_supra][isi_index]
    isi_spikes = features["spikes_set"][spikes_set_index]
    if only_one_spike:
        threshold_v = isi_spikes["threshold_v"][0]
        fast_trough_index = isi_spikes["fast_trough_index"].astype(int)[0]

        end_index = tsu.find_time_index(isi_sweep.t, isi_sweep.t[fast_trough_index] + 0.1)
        isi_raw = isi_sweep.v[fast_trough_index:end_index] - threshold_v

        width = len(isi_raw) // n_points
        isi_raw = isi_raw[:width * n_points] # ensure division will work
        isi_norm = subsample_average(isi_raw, width)

        return isi_norm

    clip_mask = ~isi_spikes["clipped"].values
    threshold_indexes = isi_spikes["threshold_index"].values[clip_mask]
    threshold_voltages = isi_spikes["threshold_v"].values[clip_mask]
    fast_trough_indexes = isi_spikes["fast_trough_index"].values[clip_mask]
    isi_list = []
    for start_index, end_index, thresh_v in zip(fast_trough_indexes[:-1], threshold_indexes[1:], threshold_voltages[:-1]):
        isi_raw = isi_sweep.v[int(start_index):int(end_index)] - thresh_v
        width = len(isi_raw) // n_points
        if width == 0:
            # trace is shorter than 100 points - probably in a burst, so we'll skip
            continue
        isi_norm = subsample_average(isi_raw[:width * n_points], width)
        isi_list.append(isi_norm)

    isi_norm = np.vstack(isi_list).mean(axis=0)
    return isi_norm


def first_ap_features(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                      lsq_features, ssq_features, ramp_features,
                      target_sampling_rate=50000, window_length=0.003):
    """Waveforms of first APs from long square, short square, and ramp

    Parameters
    ----------
    *_sweeps  : SweepSet objects
    *_features : results of *Analysis.analyze()
    target_sampling_rate : Hz
    window_length : seconds

    Returns
    -------
    output_vector : waveforms of APs
    """
    if lsq_sweeps is None and ssq_sweeps is None and ramp_sweeps is None:
        raise er.FeatureError("No input provided for first AP shape")

    # Figure out the sampling rate & target length
    if lsq_sweeps is not None:
        swp = lsq_sweeps.sweeps[0]
    elif ssq_sweeps is not None:
        swp = ssq_sweeps.sweeps[0]
    elif ramp_sweeps is not None:
        swp = ramp_sweeps.sweeps[0]
    else:
        raise er.FeatureError("Could not find any sweeps for first AP shape")
    sampling_rate = int(np.rint(1. / (swp.t[1] - swp.t[0])))
    length_in_points = int(sampling_rate * window_length)

    # Long squares
    if lsq_sweeps is not None and lsq_features is not None:
        rheo_ind = lsq_features["rheobase_sweep"].name

        sweep = lsq_sweeps.sweeps[rheo_ind]
        spikes = lsq_features["spikes_set"][rheo_ind]
        ap_long_square = first_ap_waveform(sweep, spikes, length_in_points)
    else:
        ap_long_square = np.zeros(length_in_points)

    # Ramps
    if ramp_sweeps is not None and ramp_features is not None:
        ap_ramp = np.zeros(length_in_points)
        for swp_ind in ramp_features["spiking_sweeps"].index:
            sweep = ramp_sweeps.sweeps[swp_ind]
            spikes = ramp_features["spikes_set"][swp_ind]
            ap_ramp += first_ap_waveform(sweep, spikes, length_in_points)
        if len(ramp_features["spiking_sweeps"]) > 0:
            ap_ramp /= len(ramp_features["spiking_sweeps"])
    else:
        ap_ramp = np.zeros(length_in_points)

    # Short square
    if ssq_sweeps is not None and ssq_features is not None:
        ap_short_square = np.zeros(length_in_points)
        short_count = 0
        for swp_ind in ssq_features["common_amp_sweeps"].index:
            sweep = ssq_sweeps.sweeps[swp_ind]
            spikes = ssq_features["spikes_set"][swp_ind]
            if len(spikes) > 0:
                short_count += 1
                ap_short_square += first_ap_waveform(sweep, spikes, length_in_points)
        if short_count > 0:
            ap_short_square /= short_count
    else:
        ap_short_square = np.zeros(length_in_points)

    # Downsample if necessary
    if sampling_rate > target_sampling_rate:
        sampling_factor = sampling_rate // target_sampling_rate
        ap_long_square = subsample_average(ap_long_square, sampling_factor)
        ap_ramp = subsample_average(ap_ramp, sampling_factor)
        ap_short_square = subsample_average(ap_short_square, sampling_factor)

    dv_ap_short_square = np.diff(ap_short_square)
    dv_ap_long_square = np.diff(ap_long_square)
    dv_ap_ramp = np.diff(ap_ramp)

    ap_list = [ap_short_square, ap_long_square, ap_ramp,
                               dv_ap_short_square, dv_ap_long_square, dv_ap_ramp]

    output_vector = np.hstack([ap_short_square, ap_long_square, ap_ramp,
                               dv_ap_short_square, dv_ap_long_square, dv_ap_ramp])
    return output_vector


def noise_ap_features(noise_sweeps,
                      stim_interval_list = [(2.02, 5.02), (10.02, 13.02), (18.02, 21.02)],
                      target_sampling_rate=50000, window_length=0.003,
                      skip_first_n=1):

    # Noise sweeps have three intervals of stimulation
    # so we need to find the spikes in each of them
    features_list = []
    for start, end in stim_interval_list:
        spx, spfx = dsf.extractors_for_sweeps(noise_sweeps,
                                              start=start,
                                              end=end)

        analysis = spa.StimulusProtocolAnalysis(spx, spfx)
        features_list.append(analysis.analyze(noise_sweeps))

    swp = noise_sweeps.sweeps[0]
    sampling_rate = int(np.rint(1. / (swp.t[1] - swp.t[0])))
    length_in_points = int(sampling_rate * window_length)
    avg_ap_list = []
    for i, sweep in enumerate(noise_sweeps.sweeps):
        spike_indexes = np.array([])

        # Accumulate the spike times from each interval
        # excluding the initial `skip_first_n` (due to expected systematically different shapes)
        for features in features_list:
            spikes = features["spikes_set"][i]
            if len(spikes) <= skip_first_n:
                continue

            spike_indexes = np.hstack([spike_indexes, spikes["threshold_index"].values[skip_first_n:]])

        if len(spike_indexes) > 0:
            avg_ap_list.append(avg_ap_waveform(sweep, spike_indexes, length_in_points))

    grand_avg_ap = np.vstack(avg_ap_list).mean(axis=0)
    if sampling_rate > target_sampling_rate:
        sampling_factor = sampling_rate // target_sampling_rate
        grand_avg_ap = subsample_average(grand_avg_ap, sampling_factor)

    return np.hstack([grand_avg_ap, np.diff(grand_avg_ap)])


def avg_ap_waveform(sweep, spike_indexes, length_in_points):
    avg_waveform = np.zeros(length_in_points)
    for si in spike_indexes.astype(int):
        ei = si + length_in_points
        avg_waveform += sweep.v[si:ei]

    return avg_waveform / float(len(spike_indexes))


def first_ap_waveform(sweep, spikes, length_in_points):
    start_index = spikes["threshold_index"].astype(int)[0]
    end_index = start_index + length_in_points
    return sweep.v[start_index:end_index]


def spiking_features(sweep_set, features, spike_extractor, start, end,
                     feature_width=20, rate_width=50,
                     amp_interval=20, max_above_rheo=100,
                     amp_tolerance=0.,
                     spike_feature_list=[
                        "upstroke_downstroke_ratio",
                        "peak_v",
                        "fast_trough_v",
                        "adp_delta_v",
                        "slow_trough_delta_v",
                        "slow_trough_delta_t",
                        "threshold_v",
                        "width",
                        ]):
    """Binned and interpolated per-spike features"""
    sweep_table = features["spiking_sweeps"]
    mask_supra = sweep_table["stim_amp"] >= features["rheobase_i"]
    sweep_indexes = consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
    logging.debug("Identifying spiking sweeps using rheobase = {}".format(features["rheobase_i"]))
    amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - features["rheobase_i"])
    logging.debug("Available amplitudes: {:s}".format(np.array2string(amps)))
    spike_data = features["spikes_set"]

    sweeps_to_use = spiking_sweeps_at_levels(amps, sweep_indexes,
        max_above_rheo, amp_interval, amp_tolerance)

    # We want more than one sweep to line up with expected intervals
    if len(sweeps_to_use) <= 1:
        logging.debug("Found only one spiking sweep that matches expected amplitude levels; attempting to shift by 10 pA")
        sweeps_to_use = spiking_sweeps_at_levels(amps - 10., sweep_indexes,
        max_above_rheo, amp_interval, amp_tolerance)
        if len(sweeps_to_use) <= 1:
            raise er.FeatureError("Could not find enough spiking sweeps aligned with expected amplitude levels")

    # PSTH Calculation
    rate_data = {}
    for amp_level, amp, swp_ind in sweeps_to_use:
        thresh_t = spike_data[swp_ind]["threshold_t"]
        spike_count = np.ones_like(thresh_t)
        bin_number = int(1. / 0.001) // rate_width + 1
        bins = np.linspace(start, end, bin_number)
        bin_width = bins[1] - bins[0]
        output = stats.binned_statistic(thresh_t,
                                        spike_count,
                                        statistic='sum',
                                        bins=bins)[0]
        output[np.isnan(output)] = 0
        output /= bin_width # convert to spikes/s
        rate_data[amp_level] = output

    # Combine all levels into single vector & imterpolate to fill missing sweeps
    output_vector = combine_and_interpolate(rate_data, max_level=max_above_rheo // amp_interval)

    # Instantaneous frequency
    feature_data = {}
    for amp_level, amp, swp_ind in sweeps_to_use:
        swp = sweep_set.sweeps[swp_ind]
        start_index = tsu.find_time_index(swp.t, start)
        end_index = tsu.find_time_index(swp.t, end)

        thresh_ind = spike_data[swp_ind]["threshold_index"].values
        thresh_t = spike_data[swp_ind]["threshold_t"].values
        inst_freq = inst_freq_feature(thresh_t, start, end)
        inst_freq = np.insert(inst_freq, 0, 1. / (thresh_t[0] - start))
        thresh_ind = np.insert(thresh_ind, [0, len(thresh_ind)], [start_index, end_index])

        t = swp.t[start_index:end_index]
        freq = np.zeros_like(t)
        thresh_ind -= start_index
        for f, i1, i2 in zip(inst_freq, thresh_ind[:-1], thresh_ind[1:]):
            freq[i1:i2] = f

        bin_number = int(1. / 0.001) // feature_width + 1
        bins = np.linspace(start, end, bin_number)
        output = stats.binned_statistic(t,
                                        freq,
                                        bins=bins)[0]
        feature_data[amp_level] = output
    output_vector = np.append(output_vector, combine_and_interpolate(feature_data, max_level=max_above_rheo // amp_interval))

    # Spike-level feature calculation
    for feature in spike_feature_list:
        feature_data = {}
        for amp_level, amp, swp_ind in sweeps_to_use:
            thresh_t = spike_data[swp_ind]["threshold_t"].values
            if feature not in spike_data[swp_ind].columns:
                feature_values = np.zeros_like(thresh_t)
            else:
                # Not every feature is defined for every spike
                feature_values = spike_data[swp_ind][feature].values
                can_be_clipped = spike_extractor.is_spike_feature_affected_by_clipping(feature)
                if can_be_clipped:
                    mask = ~spike_data[swp_ind]["clipped"].values
                    thresh_t = thresh_t[mask]
                    feature_values = feature_values[mask]

            bin_number = int(1. / 0.001) // feature_width + 1
            bins = np.linspace(start, end, bin_number)

            output = stats.binned_statistic(thresh_t,
                                            feature_values,
                                            bins=bins)[0]
            nan_ind = np.isnan(output)
            x = np.arange(len(output))
            output[nan_ind] = np.interp(x[nan_ind], x[~nan_ind], output[~nan_ind])
            feature_data[amp_level] = output
        output_vector = np.append(output_vector, combine_and_interpolate(feature_data, max_level=5))
    return output_vector


def spiking_sweeps_at_levels(amps, sweep_indexes, max_above_rheo, amp_interval,
        amp_tolerance):

    sweeps_to_use = []
    for amp, swp_ind in zip(amps, sweep_indexes):
        if (amp > max_above_rheo) or (amp < 0):
            logging.debug("Skipping amplitude {} (outside range)".format(amp))
            continue
        amp_level = int(np.rint(amp / float(amp_interval)))

        if (np.abs(amp - amp_level * amp_interval) > amp_tolerance):
            logging.debug("Skipping amplitude {} (mismatch with expected amplitude interval)".format(amp))
            continue

        logging.debug("Using amplitude {} for amp level {}".format(amp, amp_level))
        sweeps_to_use.append((amp_level, amp, swp_ind))
    return sweeps_to_use

def consolidated_long_square_indexes(sweep_table):
    sweep_index_list = []
    amp_arr = sweep_table["stim_amp"].unique()
    for a in amp_arr:
        ind = np.flatnonzero(sweep_table["stim_amp"] == a)
        if len(ind) == 1:
            sweep_index_list.append(sweep_table.index[ind[0]])
        else:
            # find the sweep with the median number of spikes at a given amplitude
            rates = sweep_table.iloc[ind, :]["avg_rate"].values
            median_ind = np.argmin(np.abs(np.median(rates) - rates))
            sweep_index_list.append(sweep_table.index.values[ind[median_ind]])

    return np.array(sweep_index_list)


def combine_and_interpolate(data, max_level):
    output_vector = data[0]
    if len(data) <= 1:
        logging.warning("Data from only one spiking sweep found; interpolation may have problems")
    for i in range(1, max_level + 1):
        if i not in data:
            bigger_level = -1
            for j in range(i + 1, max_level + 1):
                if j in data:
                    bigger_level = j
                    break
            smaller_level = -1
            for j in range(i - 1, -1, -1):
                if j in data:
                    smaller_level = j
                    break
            if bigger_level == -1:
                new_data = data[smaller_level]
            else:
                new_data = (data[smaller_level] + data[bigger_level]) / 2.
            output_vector = np.append(output_vector, new_data)
            data[i] = new_data
        else:
            output_vector = np.append(output_vector, data[i])
    return output_vector


def inst_freq_feature(threshold_t, start, end):
    if len(threshold_t) == 0:
        return np.array([])
    elif len(threshold_t) == 1:
        return np.array([1. / (end - start)])

    values = 1. / np.diff(threshold_t)
    values = np.append(values, 1. / (end - threshold_t[-2])) # last
    return values


