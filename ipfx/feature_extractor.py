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

from . import spike_features as spkf
from . import subthresh_features as subf
from . import spike_detector as spkd
from . import spike_train_features as strf
from . import time_series_utils as tsu


class SpikeFeatureExtractor(object):
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
                 thresh_frac=0.05, reject_at_stim_start_interval=0):
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
        reject_at_stim_start_interval : duration of window after start to reject potential spikes (optional, default 0)
        """
        self.start = start
        self.end = end
        self.filter = filter
        self.dv_cutoff = dv_cutoff
        self.max_interval = max_interval
        self.min_height = min_height
        self.min_peak = min_peak
        self.thresh_frac = thresh_frac
        self.reject_at_stim_start_interval = reject_at_stim_start_interval

    def process(self, t, v, i):
        dvdt = tsu.calculate_dvdt(v, t, self.filter)

        # Basic features of spikes
        putative_spikes = spkd.detect_putative_spikes(v, t, self.start, self.end,
                                                    dv_cutoff=self.dv_cutoff,
                                                    dvdt=dvdt)
        peaks = spkd.find_peak_indexes(v, t, putative_spikes, self.end)
        putative_spikes, peaks = spkd.filter_putative_spikes(v, t, putative_spikes, peaks,
                                                           self.min_height, self.min_peak,
                                                           dvdt=dvdt)

        if not putative_spikes.size:
            # Save time if no spikes detected
            return DataFrame()

        upstrokes = spkd.find_upstroke_indexes(v, t, putative_spikes, peaks, dvdt=dvdt)
        thresholds = spkd.refine_threshold_indexes(v, t, upstrokes, self.thresh_frac,
                                                 dvdt=dvdt)

        thresholds, peaks, upstrokes, clipped = spkd.check_thresholds_and_peaks(v, t, thresholds, peaks,
                                                                     upstrokes, self.start, self.end, self.max_interval,
                                                                     dvdt=dvdt,
                                                                     reject_at_stim_start_interval=self.reject_at_stim_start_interval)
        if not thresholds.size:
            # Save time if no spikes detected
            return DataFrame()

        # Spike list and thresholds have been refined - now find other features
        upstrokes = spkd.find_upstroke_indexes(v, t, thresholds, peaks, self.filter, dvdt)
        troughs = spkd.find_trough_indexes(v, t, thresholds, peaks, clipped, self.end)
        downstrokes = spkd.find_downstroke_indexes(v, t, peaks, troughs, clipped, dvdt=dvdt)
        trough_details, clipped = spkf.analyze_trough_details(v, t, thresholds, peaks, clipped, self.end,
                                                            dvdt=dvdt)

        widths = spkf.find_widths(v, t, thresholds, peaks, trough_details[1], clipped)


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

        # Redundant, but ensures that DataFrame has right number of rows
        # Any better way to do it?
        spikes_df = DataFrame(data=thresholds, columns=["threshold_index"])
        spikes_df["clipped"] = clipped

        for k, all_vals in vit_data_indexes.items():
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k + "_t"] = np.nan
            spikes_df[k + "_v"] = np.nan

            if len(vals) > 0:
                spikes_df.loc[valid_ind, k + "_index"] = vals
                spikes_df.loc[valid_ind, k + "_t"] = t[vals]
                spikes_df.loc[valid_ind, k + "_v"] = v[vals]

            if i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.loc[valid_ind, k + "_i"] = i[vals]

        for k, all_vals in dvdt_data_indexes.items():
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k] = np.nan
            if len(vals) > 0:
                spikes_df.loc[valid_ind, k + "_index"] = vals
                spikes_df.loc[valid_ind, k + "_t"] = t[vals]
                spikes_df.loc[valid_ind, k + "_v"] = v[vals]
                spikes_df.loc[valid_ind, k] = dvdt[vals]

        spikes_df["isi_type"] = isi_types

        for k, all_vals in trough_detail_indexes.items():
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            spikes_df[k + "_index"] = np.nan
            spikes_df[k + "_t"] = np.nan
            spikes_df[k + "_v"] = np.nan
            if len(vals) > 0:
                spikes_df.loc[valid_ind, k + "_index"] = vals
                spikes_df.loc[valid_ind, k + "_t"] = t[vals]
                spikes_df.loc[valid_ind, k + "_v"] = v[vals]

            if i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.loc[valid_ind, k + "_i"] = i[vals]

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

    def process(self, t, v, i, spikes_df, extra_features=None, exclude_clipped=False):
        features = strf.basic_spike_train_features(t, spikes_df, self.start, self.end, exclude_clipped=exclude_clipped)

        if self.start is None:
            self.start = 0.0

        if extra_features is None:
            extra_features = []

        if 'peak_deflect' in extra_features:
            features['peak_deflect'] = subf.voltage_deflection(t, v, i, self.start, self.end, self.deflect_type)

        if 'stim_amp' in extra_features:
            features['stim_amp'] = self.stim_amp_fn(t, i, self.start) if self.stim_amp_fn else None

        if 'v_baseline' in extra_features:
            features['v_baseline'] = subf.baseline_voltage(t, v, self.start, self.baseline_interval, self.filter_frequency)

        if 'sag' in extra_features:
            features['sag'] = subf.sag(t, v, i, self.start, self.end, self.peak_width, self.sag_baseline_interval)

        if features["avg_rate"] > 0:
            if 'pause' in extra_features:
                features['pause'] = strf.pause(t, spikes_df, self.start, self.end, self.pause_cost_weight)
            if 'burst' in extra_features:
                features['burst'] = strf.burst(t, spikes_df, self.burst_tol, self.pause_cost)
            if 'delay' in extra_features:
                features['delay'] = strf.delay(t, v, spikes_df, self.start, self.end)

        return features




