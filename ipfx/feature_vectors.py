import numpy as np
import logging
import pandas as pd
from scipy import stats
from . import stim_features as stf
from . import data_set_features as dsf
from . import stimulus_protocol_analysis as spa
from . import time_series_utils as tsu
import error as er


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


def extract_multipatch_feature_vectors(lsq_sweeps, lsq_start, lsq_end,
                                       target_sampling_rate=50000, ap_window_length=0.003):
    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps, start=lsq_start, end=lsq_end)
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100)
    lsq_features = lsq_an.analyze(lsq_sweeps)

    all_features = feature_vectors(lsq_sweeps, None, None,
                                   lsq_features, None, None,
                                   lsq_start, lsq_end,
                                   lsq_spx,
                                   target_sampling_rate=target_sampling_rate,
                                   ap_window_length=ap_window_length)
    return all_features


def feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                    lsq_features, ssq_features, ramp_features,
                    lsq_start, lsq_end, lsq_spike_extractor,
                    feature_width=20, rate_width=50,
                    target_sampling_rate=50000, ap_window_length=0.003):
    """Feature vectors from stimulus set features"""

    result = {}
    result["step_subthresh"] = step_subthreshold(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    result["subthresh_norm"] = subthresh_norm(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    result["isi_shape"] = isi_shape(lsq_sweeps, lsq_features)
    result["first_ap"] = first_ap_features(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                                           lsq_features, ssq_features, ramp_features,
                                           target_sampling_rate=target_sampling_rate,
                                           window_length=ap_window_length)
    result["spiking"] = spiking_features(lsq_sweeps, lsq_features, lsq_spike_extractor,
                                         lsq_start, lsq_end,
                                         feature_width, rate_width)
    return result


def step_subthreshold(sweep_set, features, start, end,
                      extend_duration=0.2, subsample_interval=0.01,
                      low_amp=-90., high_amp=10., amp_step=20.):
    """ Subsample set of subthreshold step responses including regions before and after step

        Parameters
        ----------
        sweep_set : SweepSet
        features : output of LongSquareAnalysis.analyze()
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        extend_duration : in seconds (default 0.2)
        subsample_interval : in seconds (default 0.01)

        Returns
        -------
        output_vector : subsampled, concatenated voltage trace
    """
    subthresh_df = features["subthreshold_membrane_property_sweeps"]
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweep_set.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)

    subthresh_data = {}
    for amp, swp in zip(subthresh_amps, subthresh_sweeps):
        start_index = tsu.find_time_index(swp.t, start - extend_duration)
        delta_t = swp.t[1] - swp.t[0]
        subsample_width = int(np.round(subsample_interval / delta_t))
        end_index = tsu.find_time_index(swp.t, end + extend_duration)
        subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
        subthresh_data[amp] = subsampled_v

    extend_length = int(np.round(extend_duration / subsample_interval))
    use_amps = np.arange(low_amp, high_amp, amp_step)
    n_individual = len(subsampled_v)
    neighbor_amps = sorted([a for a in subthresh_amps if a >= low_amp and a <= high_amp])
    output_vector = np.zeros(len(use_amps) * n_individual)
    for i, amp in enumerate(use_amps):
        if amp in subthresh_data:
            output_vector[i * n_individual:(i + 1) * n_individual] = subthresh_data[amp]
        else:
            lower_amp = 0
            upper_amp = 0
            for a in neighbor_amps:
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

            avg = np.zeros_like(subsampled_v)
            if lower_amp != 0 and upper_amp != 0:
                avg = (subthresh_data[lower_amp] + subthresh_data[upper_amp]) / 2.
                scale = amp / ((lower_amp + upper_amp) / 2.)
                base_v = avg[:extend_length].mean()
                avg[extend_length:-extend_length] = (avg[extend_length:-extend_length] - base_v) * scale + base_v
            elif lower_amp != 0:
                avg = subthresh_data[lower_amp].copy()
                scale = amp / lower_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            elif upper_amp != 0:
                avg = subthresh_data[upper_amp].copy()
                scale = amp / upper_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v

            output_vector[i * n_individual:(i + 1) * n_individual] = avg

    return output_vector


def subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg


def subthresh_norm(sweep_set, features, start, end,
                   extend_duration=0.2, subsample_interval=0.01):
    """ Largest amplitude subthreshold step response normalized to baseline and peak deflection

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
    subthresh_df = features["subthreshold_membrane_property_sweeps"]
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweep_set.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)

    # PRE QC ISSUE: Check for bad amps; shouldn't have to do this in general
    subthresh_amps[subthresh_amps < -1000] = np.inf

    min_sweep_ind = np.argmin(subthresh_amps)
    swp = subthresh_sweeps[min_sweep_ind]
    base = subthresh_df.at[subthresh_df.index[min_sweep_ind], "v_baseline"]
    deflect_v, deflect_ind = subthresh_df.at[subthresh_df.index[min_sweep_ind], "peak_deflect"]
    delta = base - deflect_v

    start_index = tsu.find_time_index(swp.t, start - extend_duration)
    delta_t = swp.t[1] - swp.t[0]
    subsample_width = int(np.round(subsample_interval / delta_t))
    end_index = tsu.find_time_index(swp.t, end + extend_duration)
    subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v


def isi_shape(sweep_set, features, n_points=100, min_spike=5):
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
    mask_supra = sweep_table["stim_amp"] >= features["rheobase_i"]
    amps = np.rint(sweep_table.loc[mask_supra, "stim_amp"].values - features["rheobase_i"])

    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike (use 100 ms after spike or end of trace)

    only_one_spike = False
    n_spikes = sweep_table.loc[mask_supra, "avg_rate"].values

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
    isi_spikes = np.array(features["spikes_set"])[mask_supra][isi_index]
    if only_one_spike:
        threshold_v = isi_spikes["threshold_v"][0]
        fast_trough_index = isi_spikes["fast_trough_index"].astype(int)[0]

        end_index = tsu.find_time_index(isi_sweep.t, isi_sweep.t[fast_trough_index] + 0.1)
        isi_raw = isi_sweep.v[fast_trough_index:end_index] - threshold_v

        width = len(isi_raw) / n_points
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
        width = len(isi_raw) / n_points
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
        sampling_factor = sampling_rate / target_sampling_rate
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
        sampling_factor = sampling_rate / target_sampling_rate
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
    amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - features["rheobase_i"])

    spike_data = np.array(features["spikes_set"])

    # PSTH Calculation
    rate_data = {}
    for amp, swp_ind in zip(amps, sweep_indexes):
        if (amp % amp_interval != 0) or (amp > max_above_rheo) or (amp < 0):
            continue

        amp_level = int(np.rint(amp / float(amp_interval)))
        thresh_t = spike_data[swp_ind]["threshold_t"]
        spike_count = np.ones_like(thresh_t)
        bin_number = int(1. / 0.001) / rate_width + 1
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
    output_vector = combine_and_interpolate(rate_data, max_level=max_above_rheo / amp_interval)

    # Instantaneous frequency
    feature_data = {}
    for amp, swp_ind in zip(amps, sweep_indexes):
        if (amp % amp_interval != 0) or (amp > max_above_rheo) or (amp < 0):
            continue

        amp_level = int(np.rint(amp / float(amp_interval)))

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

        bin_number = int(1. / 0.001) / feature_width + 1
        bins = np.linspace(start, end, bin_number)
        output = stats.binned_statistic(t,
                                        freq,
                                        bins=bins)[0]
        feature_data[amp_level] = output
    output_vector = np.append(output_vector, combine_and_interpolate(feature_data, max_level=max_above_rheo / amp_interval))

    # Spike-level feature calculation
    for feature in spike_feature_list:
        feature_data = {}
        for amp, swp_ind in zip(amps, sweep_indexes):
            if (amp % amp_interval != 0) or (amp > max_above_rheo) or (amp < 0):
                continue
            amp_level = int(np.rint(amp / float(amp_interval)))

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

            bin_number = int(1. / 0.001) / feature_width + 1
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


