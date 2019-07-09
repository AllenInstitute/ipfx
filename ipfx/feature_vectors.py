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


def identify_subthreshold_depol_with_amplitudes(features, sweeps):
    """ Identify subthreshold responses from depolarizing steps

        Parameters
        ----------
        features : output of LongSquareAnalysis.analyze()
        sweeps: SweepSet

        Returns:
        amp_sweep_dict : dict of sweeps with amplitudes as keys
        deflect_dict : dict of (base, deflect) tuples with amplitudes as keys
    """
    if "subthreshold_sweeps" in features:
        sweep_table = features["subthreshold_sweeps"]
    else:
        all_sweeps_table = features["sweeps"]
        sweep_table = all_sweeps_table.loc[all_sweeps_table["avg_rate"] == 0, :]

    amps = np.rint(sweep_table["stim_amp"].values)
    subthresh_sweep_ind = sweep_table.index.tolist()
    subthresh_sweeps = np.array(sweeps.sweeps)[subthresh_sweep_ind]

    subthresh_depol_mask = amps > 0
    if np.sum(subthresh_depol_mask) == 0:
        return {}, {}

    subthresh_amps = amps[subthresh_depol_mask]
    subthresh_sweeps = subthresh_sweeps[subthresh_depol_mask]
    subthresh_deflect = sweep_table["peak_deflect"].values[subthresh_depol_mask]
    subthresh_base = sweep_table["v_baseline"].values[subthresh_depol_mask]

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
        subsampled_v = _subsample_average(swp.v[start_index:end_index], subsample_width)
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


def _subsample_average(x, width):
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
    subsampled_v = _subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v


def subthresh_depol_norm(amp_sweep_dict, deflect_dict, start, end,
    extend_duration=0.2, subsample_interval=0.01, steady_state_interval=0.1):
    """ Largest positive-going subthreshold step response that does not evoke spikes,
        normalized to baseline and steady-state at end of step

        Parameters
        ----------
        amp_sweep_dict : dict of amplitude / sweep pairs
        deflect_dict : dict of (baseline, deflect) tuples with amplitude keys
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        extend_duration: in seconds (default 0.2)
        subsample_interval: in seconds (default 0.01)
        steady_state_interval: interval before end for normalization (seconds, default=0.1)

        Returns
        -------
        subsampled_v : subsampled, normalized voltage trace
    """

    if (end - start) < steady_state_interval:
        raise ValueError("steady state interval cannot exceed stimulus interval")

    if len(amp_sweep_dict) == 0:
        logging.debug("No subthreshold depolarizing sweeps found - returning all-nan response")

        # create all-nan response of appropriate length
        total_interval = extend_duration * 2 + (end - start)
        length = int(total_interval / subsample_interval)
        return np.ones(length) * np.nan

    available_amps = list(amp_sweep_dict.keys())
    max_amp = np.max(available_amps)
    swp = amp_sweep_dict[max_amp]

    base, _ = deflect_dict[max_amp]

    interval_start_index = tsu.find_time_index(swp.t, end - steady_state_interval)
    interval_end_index = tsu.find_time_index(swp.t, end)
    steady_state_v = swp.v[interval_start_index:interval_end_index].mean()

    delta = steady_state_v - base

    start_index = tsu.find_time_index(swp.t, start - extend_duration)
    delta_t = swp.t[1] - swp.t[0]
    subsample_width = int(np.round(subsample_interval / delta_t))
    end_index = tsu.find_time_index(swp.t, end + extend_duration)
    subsampled_v = _subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v

def identify_sweep_for_isi_shape(sweeps, features, duration, min_spike=5):
    """ Find lowest-amplitude spiking sweep that has at least min_spike
        or else sweep with most spikes

        Parameters
        ----------
        sweeps: SweepSet
        features: output of LongSquareAnalysis.analyze()
        duration: length of stimulus interval (seconds)
        min_spike: minimum number of spikes for first preference sweep (default 5)

        Returns
        -------
        selected_sweep : Sweep
        selected_spike_info : DataFrame of spike info for selected sweep
    """
    sweep_table = features["sweeps"]
    mask_supra = (sweep_table["avg_rate"].values > 0) & (sweep_table["stim_amp"] > 0)
    supra_table = sweep_table.loc[mask_supra, :]
    amps = np.rint(supra_table["stim_amp"].values)
    n_spikes = supra_table["avg_rate"].values * duration

    # Pick out the sweep to get the ISI shape
    # Shape differences are more prominent at lower frequencies, but we want
    # enough to average to reduce noise. So, we will pick
    # (1) lowest amplitude sweep with at least `min_spike` spikes (i.e. min_spike - 1 ISIs)
    # (2) if not (1), sweep with the most spikes if any have multiple spikes
    # (3) if not (1) or (2), lowest amplitude sweep with 1 spike (use 100 ms after spike or end of trace)

    if np.any(n_spikes >= min_spike):
        spike_mask = n_spikes >= min_spike
        min_index = np.argmin(amps[spike_mask])
        selection_index = np.arange(0, len(n_spikes), dtype=int)[spike_mask][min_index]
    elif np.any(n_spikes > 1):
        selection_index = np.argmax(n_spikes)
    else:
        only_one_spike = True
        selection_index = np.argmin(amps)

    selected_sweep = np.array(sweeps.sweeps)[mask_supra][selection_index]
    info_index = supra_table.index.tolist()[selection_index]
    selected_spike_info = features["spikes_set"][info_index]
    return selected_sweep, selected_spike_info


def isi_shape(sweep, spike_info, end, n_points=100, steady_state_interval=0.1,
    single_return_tolerance=1., single_max_duration=0.1):
    """ Average interspike voltage trajectory with normalized duration, aligned to threshold

        Parameters
        ----------
        sweep : Sweep
        spike_info : dataframe of spike info for sweep
        n_points: number of points in output
        end : stimulus interval end (seconds)
        steady_state_interval: interval for calculating steady-state for
            sweeps with only one spike (default 0.1 s)
        single_return_tolerance: allowable difference from steady-state for
            determining end of "ISI" if only one spike is in sweep (default 1 mV)
        single_max_duration: allowable max duration for finding end of "ISI"
            if only one spike is in sweep (default 0.1 s)

        Returns
        -------
        isi_norm : averaged, threshold-aligned, and duration-normalized voltage trace
    """

    n_spikes = spike_info.shape[0]

    if n_spikes > 1:
        threshold_indexes = spike_info["threshold_index"].values
        threshold_voltages = spike_info["threshold_v"].values
        fast_trough_indexes = spike_info["fast_trough_index"].values
        isi_list = []
        for start_index, end_index, thresh_v in zip(fast_trough_indexes[:-1], threshold_indexes[1:], threshold_voltages[:-1]):
            isi_raw = sweep.v[int(start_index):int(end_index)] - thresh_v
            width = len(isi_raw) // n_points
            if width == 0:
                logging.debug("found isi shorter than specified width; skipping")
                continue
            isi_norm = _subsample_average(isi_raw[:width * n_points], width)
            isi_list.append(isi_norm)

        isi_norm = np.vstack(isi_list).mean(axis=0)
    else:
        threshold_v = spike_info["threshold_v"][0]
        fast_trough_index = spike_info["fast_trough_index"].astype(int)[0]
        fast_trough_t = spike_info["fast_trough_t"][0]
        stim_end_index = tsu.find_time_index(sweep.t, end)
        if fast_trough_t < end - steady_state_interval:
            max_end_index = tsu.find_time_index(sweep.t, sweep.t[fast_trough_index] + single_max_duration)

            std_start_index = tsu.find_time_index(sweep.t, end - steady_state_interval)
            steady_state_v = sweep.v[std_start_index:stim_end_index].mean()

            above_ss_ind = np.flatnonzero(sweep.v[fast_trough_index:] >= steady_state_v - single_return_tolerance)

            if len(above_ss_ind) > 0:
                end_index = above_ss_ind[0] + fast_trough_index
                end_index = min(end_index, max_end_index)
            else:
                logging.debug("isi_shape: voltage does not return to steady-state within specified tolerance; "
                    "resorting to specified max duration")
                end_index = max_end_index
        else:
            end_index = stim_end_index

        isi_raw = sweep.v[fast_trough_index:end_index] - threshold_v
        width = len(isi_raw) // n_points
        isi_raw = isi_raw[:width * n_points] # ensure division will work
        isi_norm = _subsample_average(isi_raw, width)

    return isi_norm


def first_ap_vectors(sweeps_list, spike_info_list,
        target_sampling_rate=50000, window_length=0.003):
    """Average waveforms of first APs from sweeps

    Parameters
    ----------
    sweeps_list: list of Sweeps
    spike_info_list: list of spike info DataFrames
    target_sampling_rate : Hz
    window_length : seconds

    Returns
    -------
    ap_v: waveform of average APs
    ap_dv: waveform of first derivative of ap_v
    """

    if len(sweeps_list) == 0:
        length_in_points = int(target_sampling_rate * window_length)
        zero_v = np.zeros(length_in_points)
        return zero_v, np.diff(zero_v)

    swp = sweeps_list[0]
    sampling_rate = int(np.rint(1. / (swp.t[1] - swp.t[0])))
    length_in_points = int(sampling_rate * window_length)

    ap_list = []
    for swp, si in zip(sweeps_list, spike_info_list):
        ap = first_ap_waveform(swp, si, length_in_points)
        ap_list.append(ap)

    avg_ap = np.vstack(ap_list).mean(axis=0)

    if sampling_rate > target_sampling_rate:
        sampling_factor = int(sampling_rate // target_sampling_rate)
        avg_ap = _subsample_average(avg_ap, sampling_factor)

    return avg_ap, np.diff(avg_ap)


def noise_ap_features(noise_sweeps,
                      stim_interval_list = [(2.02, 5.02), (10.02, 13.02), (18.02, 21.02)],
                      target_sampling_rate=50000, window_length=0.003,
                      skip_first_n=1):
    """Average AP waveforms in noise sweeps

    Parameters
    ----------
    noise_sweeps: SweepSet
    stim_interval_list: list of tuples of start and end times (in seconds)
        of analysis intervals
    target_sampling_rate: Hz (default 50000)
    window_length: seconds (default 0.003)
    skip_first_n: number of initial APs to exclude from average (default 1)

    Returns
    -------
    ap_v: waveform of average AP
    ap_dv: waveform of first derivative of ap_v
    """
    # Noise sweeps have multiple intervals of stimulation
    # so we can find the spikes in each of them separately
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
            avg_ap_list.append(_avg_ap_waveform(sweep, spike_indexes, length_in_points))

    grand_avg_ap = np.vstack(avg_ap_list).mean(axis=0)
    if sampling_rate > target_sampling_rate:
        sampling_factor = sampling_rate // target_sampling_rate
        grand_avg_ap = _subsample_average(grand_avg_ap, sampling_factor)

    return grand_avg_ap, np.diff(grand_avg_ap)


def _avg_ap_waveform(sweep, spike_indexes, length_in_points):
    """ Average together spike waveforms in sweep found at spike_indexes"""

    avg_list = [sweep.v[si:si + length_in_points]
        for si in spike_indexes.astype(int)]
    return np.vstack(avg_list).mean(axis=0)


def first_ap_waveform(sweep, spikes, length_in_points):
    "Waveform of first AP with `length_in_points` time samples"

    start_index = spikes["threshold_index"].astype(int)[0]
    end_index = start_index + length_in_points
    return sweep.v[start_index:end_index]


def identify_suprathreshold_sweep_sequence(features, target_amplitudes,
        shift=None, amp_tolerance=0):
    """ Find sweeps matching desired amplitudes relative to rheobase

    Parameters
    ----------
    features: output of LongSquareAnalysis.analyze()
    target_amplitudes: list of amplitudes (relative to rheobase) for each desired step
    amp_tolerance: float, pA. Tolerance for matching amplitude (default 0)
    shift: optional, float in pA for amount to consider shifting "rheobase" to identify more
        matching sweeps if only a single sweep matches. A value of None means that no shift is attempted.

    Returns
    -------
    info_list: Spike info in order of desired amplitudes. If a given amplitude cannot be found,
        the list has `None` at that location
    """

    sweep_table = features["spiking_sweeps"]

    mask_supra = sweep_table["stim_amp"] >= features["rheobase_i"]
    sweep_table = sweep_table.loc[mask_supra, :]

    sweep_indexes = _consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
    sweep_table = sweep_table.loc[sweep_indexes, :]

    logging.debug("Identifying spiking sweeps using rheobase = {}".format(features["rheobase_i"]))
    amps = np.rint(sweep_table["stim_amp"].values - features["rheobase_i"])
    logging.debug("Available amplitudes: {:s}".format(np.array2string(amps)))
    logging.debug("Target amplitudes: {:s}".format(np.array2string(target_amplitudes)))

    spike_data = features["spikes_set"]
    sweeps_to_use = _spiking_sweeps_at_levels(amps, sweep_indexes,
        target_amplitudes, amp_tolerance)
    n_matches = np.sum([s is not None for s in sweeps_to_use])

    if len(target_amplitudes) > 1 and n_matches <= 1 and shift is not None:
        logging.debug("Found only one spiking sweep that matches expected amplitude levels; attempting to shift by {} pA".format(shift))
        sweeps_to_use = _spiking_sweeps_at_levels(amps - shift, sweep_indexes,
            target_amplitudes, amp_tolerance)
        n_matches = np.sum([s is not None for s in sweeps_to_use])

    if len(target_amplitudes) > 1 and n_matches <= 1:
        raise er.FeatureError("Could not find at least two spiking sweeps matching requested amplitude levels")

    return [spike_data[ind] if ind is not None else None for ind in sweeps_to_use]


def psth_vector(spike_info_list, start, end, width=50):
    """ Create binned "PSTH"-like feature vector based on spike times, concatenated
        across sweeps

    Parameters
    ----------
    spike_info_list: list of spike info DataFrames for each sweep
    start : stimulus interval start (seconds)
    end : stimulus interval end (seconds)
    width: bin width in ms (default 50)

    Returns
    -------
    output: Concatenated vector of binned spike rates
    """

    vector_list = []
    for si in spike_info_list:
        if si is None:
            vector_list.append(None)
            continue
        thresh_t = si["threshold_t"]
        spike_count = np.ones_like(thresh_t)
        one_ms = 0.001
        duration = end - start
        n_bins = int(duration / one_ms) // width
        bin_edges = np.linspace(start, end, n_bins + 1) # includes right edge, so adding one to desired bin number
        bin_width = bin_edges[1] - bin_edges[0]
        output = stats.binned_statistic(thresh_t,
                                        spike_count,
                                        statistic='sum',
                                        bins=bin_edges)[0]
        output[np.isnan(output)] = 0
        output /= bin_width # convert to spikes/s
        vector_list.append(output)
    output_vector = _combine_and_interpolate(vector_list)
    return output_vector


def inst_freq_vector(spike_info_list, start, end, width=20):
    """ Create binned instantaneous frequency feature vector,
        concatenated across sweeps

    Parameters
    ----------
    spike_info_list: list of spike info DataFrames for each sweep
    start : stimulus interval start (seconds)
    end : stimulus interval end (seconds)
    width: bin width in ms (default 20)

    Returns
    -------
    output: Concatenated vector of binned spike rates
    """

    vector_list = []
    for si in spike_info_list:
        if si is None:
            vector_list.append(None)
            continue
        thresh_t = si["threshold_t"].values
        inst_freq, inst_freq_times = _inst_freq_feature(thresh_t, start, end)

        one_ms = 0.001
        duration = end - start
        n_bins = int(duration / one_ms) // width
        bin_edges = np.linspace(start, end, n_bins + 1) # includes right edge, so adding one to desired bin number
        bin_width = bin_edges[1] - bin_edges[0]

        output = stats.binned_statistic(inst_freq_times,
                                        inst_freq,
                                        bins=bin_edges)[0]
        nan_ind = np.isnan(output)
        x = np.arange(len(output))
        output[nan_ind] = np.interp(x[nan_ind], x[~nan_ind], output[~nan_ind])
        vector_list.append(output)

    output_vector = _combine_and_interpolate(vector_list)

    return output_vector


def spike_feature_vector(feature, spike_info_list, start, end, width=20):
    """ Create binned feature vector for specified features,
        concatenated across sweeps

    Parameters
    ----------
    feature: string, name of feature found in members of spike_info_list
    spike_info_list: list of spike info DataFrames for each sweep
    start : stimulus interval start (seconds)
    end : stimulus interval end (seconds)
    width: bin width in ms (default 20)

    Returns
    -------
    output: Concatenated vector of binned spike rates
    """

    vector_list = []
    for si in spike_info_list:
        if si is None:
            vector_list.append(None)
            continue
        thresh_t = si["threshold_t"].values
        if feature not in si.columns:
            logging.warning("Requested feature {} not found in supplied dataframe".format(feature))
            feature_values = np.zeros_like(thresh_t)
        else:
            feature_values = si[feature].values
            mask = ~si["clipped"].values
            thresh_t = thresh_t[mask]
            feature_values = feature_values[mask]

        one_ms = 0.001
        duration = end - start
        n_bins = int(duration / one_ms) // width
        bin_edges = np.linspace(start, end, n_bins + 1) # includes right edge, so adding one to desired bin number
        bin_width = bin_edges[1] - bin_edges[0]

        output = stats.binned_statistic(thresh_t,
                                        feature_values,
                                        bins=bin_edges)[0]
        nan_ind = np.isnan(output)
        x = np.arange(len(output))
        output[nan_ind] = np.interp(x[nan_ind], x[~nan_ind], output[~nan_ind])
        vector_list.append(output)

    output_vector = _combine_and_interpolate(vector_list)
    return output_vector


def _spiking_sweeps_at_levels(amps, sweep_indexes, target_amplitudes,
        amp_tolerance):
    """Search for sweep indexes that match target amplitudes"""

    sweeps_to_use = []
    for target_amp in target_amplitudes:
        found_match = False
        for amp, swp_ind in zip(amps, sweep_indexes):
            if (np.abs(amp - target_amp) <= amp_tolerance):
                found_match = True
                sweeps_to_use.append(swp_ind)
                logging.debug("Using amplitude {} for target {}".format(amp, target_amp))
                break
        if not found_match:
            sweeps_to_use.append(None)

    return sweeps_to_use


def _consolidated_long_square_indexes(sweep_table):
    """Identify a single sweep for each stimulus amplitude if an amplitude
        is repeated
    """

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


def _combine_and_interpolate(data):
    """Concatenate and interpolate missing items from neighbors"""

    n_populated = np.sum([d is not None for d in data])
    if n_populated <= 1 and len(data) > 1:
        logging.warning("Data from only one spiking sweep found; interpolated sweeps may have issues")
    output_list = []
    for i, d in enumerate(data):
        if d is not None:
            output_list.append(d)
            continue

        # Missing data is interpolated from neighbors in list
        lower = -1
        upper = np.inf
        for j, neighbor in enumerate(data):
            if j == i:
                continue
            if neighbor is not None:
                if j < i and j > lower:
                    lower = j
                if j > i and j < upper:
                    upper = j
        if lower > -1 and upper < len(data):
            new_data = (data[lower] + data[upper]) / 2
        elif lower == -1:
            new_data = data[upper]
        else:
            new_data = data[lower]

        output_list.append(new_data)

    return np.hstack(output_list)


def _inst_freq_feature(threshold_t, start, end):
    """ Estimate instantaneous firing rate from differences in spike times

    This function attempts to estimate a semi-continuous instantanteous firing
    rate from a set of interspike intervals (ISIs) and spike times. It makes
    several assumptions:
    - It assumes that the instantaneous firing rate at the start of the stimulus
    interval is the inverse of the latency to the first spike.
    - It estimates the firing rate at each spike as the average of the ISIs
    on each side of the spike
    - If the time between the end of the interval and the last spike is less
    than the last true interspike interval, it sets the instantaneous rate of
    that last spike and of the end of the interval to the inverse of the last ISI.
    Therefore, the instantaneous rate would not "jump" just because the
    stimulus interval ends.
    - However, if the time between the end of the interval and the last spike is
    longer than the final ISI, it assumes there may have been a spike just
    after the end of the interval. Therefore, it essentially returns an upper
    bound on the estimated rate.


    Parameters
    ----------
    threshold_t: array
        Spike times
    start: float
        start of stimulus interval (seconds)
    end: float
        end of stimulus interval (seconds)

    Returns
    -------
    inst_firing_rate: array
        Instantaneous firing rate estimates (spikes/s)
    time_points: array
        Time points for corresponding values in inst_firing_rate
    """
    if len(threshold_t) == 0:
        return np.array([0, 0]), np.array([start, end])
    inst_inv_rate = []
    time_points = []
    isis = [(threshold_t[0] - start)] + np.diff(threshold_t).tolist()
    isis = isis + [max(isis[-1], end - threshold_t[-1])]

    # Estimate at start of stimulus interval
    inst_inv_rate.append(isis[0])
    time_points.append(start)

    # Estimate for each spike time
    for t, pre_isi, post_isi in zip(threshold_t, isis[:-1], isis[1:]):
        inst_inv_rate.append((pre_isi + post_isi) / 2)
        time_points.append(t)

    # Estimate for end of stimulus interval
    inst_inv_rate.append(isis[-1])
    time_points.append(end)


    inst_firing_rate = 1 / np.array(inst_inv_rate)
    time_points = np.array(time_points)
    return inst_firing_rate, time_points


