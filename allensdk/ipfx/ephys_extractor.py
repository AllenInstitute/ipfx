# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2015-2016. Allen Institute. All rights reserved.
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
from pandas import DataFrame
import warnings
import logging

from . import ephys_features as ft
import six

class SpikeExtractor(object):
    AFFECTED_BY_CLIPPING = [
        "trough_t", "trough_v", "trough_i", "trough_index",
        "downstroke", "downstroke_t","downstroke_v", "downstroke_index",
        "fast_trough_t", "fast_trough_v", "fast_trough_i", "fast_trough_index"
        "adp_t", "adp_v", "adp_i", "adp_index",
        "slow_trough_t", "slow_trough_v", "slow_trough_i", "slow_trough_index",
        "isi_type", "width", "upstroke_downstroke_ratio" ]

    """Feature calculation for a sweep (voltage and/or current time series)."""

    def __init__(self, start=None, end=None, filter=10.,
                 dv_cutoff=20., max_interval=0.005, min_height=2., min_peak=-30.,
                 thresh_frac=0.05):
        """Initialize SweepFeatures object.-

        Parameters
        ----------
        t : ndarray of times (seconds)
        v : ndarray of voltages (mV)
        i : ndarray of currents (pA)
        start : start of time window for feature analysis (optional)
        end : end of time window for feature analysis (optional)
        filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
        dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
        max_interval : maximum acceptable time between start of spike and time of peak in sec (optional, default 0.005)
        min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
        min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
        thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
        """
        self.start = start
        self.end = end
        self.filter = filter
        self.dv_cutoff = dv_cutoff
        self.max_interval = max_interval
        self.min_height = min_height
        self.min_peak = min_peak
        self.thresh_frac = thresh_frac

    def process(self, t, v, i):
        dvdt = ft.calculate_dvdt(v, t, self.filter)

        # Basic features of spikes
        putative_spikes = ft.detect_putative_spikes(v, t, self.start, self.end,
                                                    self.filter, self.dv_cutoff)
        peaks = ft.find_peak_indexes(v, t, putative_spikes, self.end)
        putative_spikes, peaks = ft.filter_putative_spikes(v, t, putative_spikes, peaks,
                                                           self.min_height, self.min_peak)

        if not putative_spikes.size:
            # Save time if no spikes detected
            return DataFrame()

        upstrokes = ft.find_upstroke_indexes(v, t, putative_spikes, peaks, self.filter, dvdt)
        thresholds = ft.refine_threshold_indexes(v, t, upstrokes, self.thresh_frac,
                                                 self.filter, dvdt)
        thresholds, peaks, upstrokes, clipped = ft.check_thresholds_and_peaks(v, t, thresholds, peaks,
                                                                     upstrokes, self.end, self.max_interval)
        if not thresholds.size:
            # Save time if no spikes detected
            return DataFrame()

        # Spike list and thresholds have been refined - now find other features
        upstrokes = ft.find_upstroke_indexes(v, t, thresholds, peaks, self.filter, dvdt)
        troughs = ft.find_trough_indexes(v, t, thresholds, peaks, clipped, self.end)
        downstrokes = ft.find_downstroke_indexes(v, t, peaks, troughs, clipped, self.filter, dvdt)
        trough_details, clipped = ft.analyze_trough_details(v, t, thresholds, peaks, clipped, self.end,
                                                            self.filter, dvdt=dvdt)

        widths = ft.find_widths(v, t, thresholds, peaks, trough_details[1], clipped)


        # Points where we care about t, v, and i if available
        vit_data_indexes = {
            "threshold": thresholds,
            "peak": peaks,
            "trough": troughs,
        }

        # Points where we care about t and dv/dt
        dvdt_data_indexes = {
            "upstroke": upstrokes,
            "downstroke": downstrokes
        }

        # Trough details
        isi_types = trough_details[0]
        trough_detail_indexes = dict(zip(["fast_trough", "adp", "slow_trough"], trough_details[1:]))
#        print "trough_detail_indexes:", trough_detail_indexes

        # Redundant, but ensures that DataFrame has right number of rows
        # Any better way to do it?
        spikes_df = DataFrame(data=thresholds, columns=["threshold_index"])
        spikes_df["clipped"] = clipped

        for k, all_vals in six.iteritems(vit_data_indexes):
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k + "_t"] = np.nan
            spikes_df[k + "_v"] = np.nan

            if len(vals) > 0:
                spikes_df.ix[valid_ind, k + "_index"] = vals
                spikes_df.ix[valid_ind, k + "_t"] = t[vals]
                spikes_df.ix[valid_ind, k + "_v"] = v[vals]

            if i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.ix[valid_ind, k + "_i"] = i[vals]

        for k, all_vals in six.iteritems(dvdt_data_indexes):
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k] = np.nan
            if len(vals) > 0:
                spikes_df.ix[valid_ind, k + "_index"] = vals
                spikes_df.ix[valid_ind, k + "_t"] = t[vals]
                spikes_df.ix[valid_ind, k + "_v"] = v[vals]
                spikes_df.ix[valid_ind, k] = dvdt[vals]

        spikes_df["isi_type"] = isi_types

        for k, all_vals in six.iteritems(trough_detail_indexes):
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k + "_t"] = np.nan
            spikes_df[k + "_v"] = np.nan
            if len(vals) > 0:
                spikes_df.ix[valid_ind, k + "_index"] = vals
                spikes_df.ix[valid_ind, k + "_t"] = t[vals]
                spikes_df.ix[valid_ind, k + "_v"] = v[vals]

            if i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.ix[valid_ind, k + "_i"] = i[vals]

        spikes_df["width"] = widths

        spikes_df["upstroke_downstroke_ratio"] = spikes_df["upstroke"] / -spikes_df["downstroke"]

        return spikes_df

    def spikes(self, spikes_df):
        """Get all features for each spike as a list of records."""
        return spikes_df.to_dict(orient='records')

    def is_spike_feature_affected_by_clipping(self, key):
        return key in self.AFFECTED_BY_CLIPPING

    def spike_feature_keys(self, spikes_df):
        """Get list of every available spike feature."""
        return spikes_df.columns.values.tolist()

    def spike_feature(self, spikes_df, key, include_clipped=False, force_exclude_clipped=False):
        """Get specified feature for every spike.

        Parameters
        ----------
        key : feature name
        include_clipped: return values for every identified spike, even when clipping means they will be incorrect/undefined

        Returns
        -------
        spike_feature_values : ndarray of features for each spike
        """

        if len(spikes_df) == 0:
            return np.array([])

        if key not in spikes_df.columns:
            raise KeyError("requested feature '{:s}' not available".format(key))

        values = spikes_df[key].values

        if include_clipped and force_exclude_clipped:
            raise ValueError("include_clipped and force_exclude_clipped cannot both be true")

        if not include_clipped and self.is_spike_feature_affected_by_clipping(key):
            values = values[~spikes_df["clipped"].values]
        elif force_exclude_clipped:
            values = values[~spikes_df["clipped"].values]

        return values

class SpikeTrainFeatureExtractor(object):
    def __init__(self, start, end,
                #pause_cost_weight=1.0,
                 burst_tol=0.5, pause_cost=1.0,
                 #deflect_type="min",
                 deflect_type=None,
                 stim_amp_fn=None,
                 baseline_interval=0.1, filter_frequency=1.0,
                 sag_baseline_interval=0.03,
                 peak_width=0.005):
        self.start = start
        self.end = end
        self.burst_tol = burst_tol
        self.pause_cost = pause_cost
        #self.pause_cost_weight = pause_cost_weight
        self.deflect_type = deflect_type
        self.stim_amp_fn = stim_amp_fn
        self.baseline_interval = baseline_interval
        self.filter_frequency = filter_frequency
        self.sag_baseline_interval = sag_baseline_interval
        self.peak_width = peak_width

    def process(self, t, v, i, spikes_df, extra_features=None):
        features = basic_spike_train_features(t, spikes_df, self.start, self.end)

        if extra_features is None:
            extra_features = []

        if 'peak_deflect' in extra_features:
            features['peak_deflect'] = ft.voltage_deflection(t, v, i, self.start, self.end, self.deflect_type)

        if 'stim_amp' in extra_features:
            features['stim_amp'] = self.stim_amp_fn(t, i, self.start) if self.stim_amp_fn else None

        if 'v_baseline' in extra_features:
            features['v_baseline'] = ft.baseline_voltage(t, v, self.start, self.baseline_interval, self.filter_frequency)

        if 'sag' in extra_features:
            features['sag'] = ft.sag(t, v, i, self.start, self.end, self.peak_width, self.sag_baseline_interval)

        if features["avg_rate"] > 0:
            if 'pause' in extra_features:
                features['pause'] = pause(t, spikes_df, self.start, self.end, self.pause_cost_weight)
            if 'burst' in extra_features:
                features['burst'] = burst(t, spikes_df, self.burst_tol, self.pause_cost)
            if 'delay' in extra_features:
                features['delay'] = delay(t, v, spikes_df, self.start, self.end)

        return features


def basic_spike_train_features(t, spikes_df, start, end):
    features = {}
    if len(spikes_df) == 0 or spikes_df.empty:
        features["avg_rate"] = 0
        return features

    thresholds = spikes_df["threshold_index"].values.astype(int)
    isis = ft.get_isis(t, thresholds)
    with warnings.catch_warnings():
        # ignore mean of empty slice warnings here
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

        features = {
            "adapt": ft.adaptation_index(isis),
            "latency": ft.latency(t, thresholds, start),
            "isi_cv": (isis.std() / isis.mean()) if len(isis) >= 1 else np.nan,
            "mean_isi": isis.mean() if len(isis) > 0 else np.nan,
            "median_isi": np.median(isis),
            "first_isi": isis[0] if len(isis) >= 1 else np.nan,
            "avg_rate": ft.average_rate(t, thresholds, start, end),
        }

    return features

def pause(self, t, spikes_df, start, end, cost_weight=1.0):
    """Estimate average number of pauses and average fraction of time spent in a pause

    Attempts to detect pauses with a variety of conditions and averages results together.

    Pauses that are consistently detected contribute more to estimates.

    Returns
    -------
    avg_n_pauses : average number of pauses detected across conditions
    avg_pause_frac : average fraction of interval (between start and end) spent in a pause
    max_reliability : max fraction of times most reliable pause was detected given weights tested
    n_max_rel_pauses : number of pauses detected with `max_reliability`
    """
    warnings.warn("This function will be removed")
    # Pauses are unusually long ISIs with a "detour reset" among delay resets
    thresholds = spikes_df["threshold_index"].values.astype(int)
    isis = ft.get_isis(t, thresholds)
    isi_types = spikes_df["isi_type"][:-1].values

    pause_list = ft.detect_pauses(isis, isi_types, cost_weight)

    if len(pause_list) == 0:
        return 0, 0.

    n_pauses = len(pause_list)
    pause_frac = isis[pause_list].sum()
    pause_frac /= end - start

    return n_pauses, pause_frac


def burst(t, spikes_df, tol=0.5, pause_cost=1.0):
    """Find bursts and return max "burstiness" index (normalized max rate in burst vs out).

    Returns
    -------
    max_burstiness_index : max "burstiness" index across detected bursts
    num_bursts : number of bursts detected
    """
    warnings.warn("This function will be removed")
    thresholds = spikes_df["threshold_index"].values.astype(int)
    isis = ft.get_isis(t, thresholds)

    isi_types = spikes_df["isi_type"][:-1].values
    fast_tr_v = spikes_df["fast_trough_v"].values
    fast_tr_t = spikes_df["fast_trough_t"].values
    slow_tr_v = spikes_df["slow_trough_v"].values
    slow_tr_t = spikes_df["slow_trough_t"].values
    thr_v = spikes_df["threshold_v"].values

    bursts = ft.detect_bursts(isis, isi_types,
                              fast_tr_v, fast_tr_t,
                              slow_tr_v, slow_tr_t,
                              thr_v, tol, pause_cost)

    burst_info = np.array(bursts)

    if burst_info.shape[0] > 0:
        return burst_info[:, 0].max(), burst_info.shape[0]
    else:
        return 0., 0

def delay(t, v, spikes_df, start, end):
    """Calculates ratio of latency to dominant time constant of rise before spike

    Returns
    -------
    delay_ratio : ratio of latency to tau (higher means more delay)
    tau : dominant time constant of rise before spike
    """
    warnings.warn("This function will be removed")

    if len(spikes_df) == 0:
        logging.info("No spikes available for delay calculation")
        return 0., 0.

    spike_time = spikes_df["threshold_t"].values[0]

#    tau = ft.fit_prespike_time_constant(t, v, start, spike_time)
    latency = spike_time - start

    delay_ratio = latency / tau
    return delay_ratio, tau

def fit_fi_slope(stim_amps, avg_rates):
    """Fit the rate and stimulus amplitude to a line and return the slope of the fit."""

    if len(stim_amps) < 2:
        raise ft.FeatureError("Cannot fit f-I curve slope with less than two sweeps")

    x = stim_amps
    y = avg_rates

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y)[0]

    return m




