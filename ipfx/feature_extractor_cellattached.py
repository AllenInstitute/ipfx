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

    """Feature calculation for a cell-attached sweep current time series."""

    def __init__(self, start=None, end=None, filter=2.,
                 di_cutoff=40., max_interval=0.005, min_peak=-15.,
                 baseline_win=0.01, fraction=0.8, time_win=0.001, 
                 thresh_frac=0.05, reject_at_stim_start_interval=0):
        """Initialize SweepFeatures object.-

        Parameters
        ----------
        t : ndarray of times (seconds)
        i : ndarray of currents (pA)
        i_baseline_filt : i that has been super low-threshold filtered, for baseline stability 
        v : ndarray of voltages (mV)
        sample_freq : sampling frequency
        start : start of time window for feature analysis (optional)
        end : end of time window for feature analysis (optional)
        filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 2)
        di_cutoff : minimum dI/dt to qualify as a spike in I/s (optional, default 40)
        max_interval : maximum acceptable time between start of spike and time of peak in sec (optional, default 0.005)
        min_peak : minimum acceptable peak level (optional, default -15)
        baseline_win : what time window (s) after putative spike to consider for local baseline measurement (optional, default 0.01 s)
        fraction : what fraction of negative spike amplitude to consider as threshold for removing biphasic (non-biological) spikes (optional, default 0.8)
        time_win : what time window (s) around putative spike to consider when looking for biphasic counterpart (optional, default 0.001 s)
        thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
        reject_at_stim_start_interval : duration of window after start to reject potential spikes (optional, default 0)
        """
        self.start = start
        self.end = end
        self.filter = filter
        self.di_cutoff = di_cutoff
        self.max_interval = max_interval
        self.min_peak = min_peak
        self.baseline_win = baseline_win
        self.fraction = fraction
        self.time_win = time_win
        self.thresh_frac = thresh_frac
        self.reject_at_stim_start_interval = reject_at_stim_start_interval

    def process(self, t, i, v, sampling_rate, i_baseline_filt):
        # (Using the same function as for iclamp here, but passing current instead of voltage)
        didt = tsu.calculate_dvdt(i, t, self.filter)

        # Basic features of spikes
        putative_spikes = spkd.detect_putative_spikes(i, t, self.start, self.end,
                                                    dv_cutoff=self.di_cutoff,
                                                    dvdt=didt)
       
        # (Using the same function as for iclamp here, but passing current instead of voltage)
        # (also flipping the signal, since we are looking for a negative peak)
        putative_peaks = spkd.find_peak_indexes(-i, t, putative_spikes, self.end)
        
       
        # Refine spike times
        #refined_peaks = spkd.filter_putative_peaks_cellattached(i, i_baseline_filt, t, sampling_rate, putative_peaks,
        #                                                min_peak=self.min_peak, baseline_win=self.baseline_win, 
        #                                                fraction=self.fraction, time_win=self.time_win, didt=didt)


        #if not refined_peaks.size:
        #    # Save time if no spikes detected
        #     return DataFrame(), DataFrame(data=putative_peaks, columns=["putative_peaks_index"])
        
        # Spike list and thresholds have been refined - now find other features
        
        # Points where we care about t and i
        it_data_indexes = {
            "peak": putative_peaks
        }

        # Redundant, but ensures that DataFrame has right number of rows
        # Any better way to do it?
        putative_peaks_df = DataFrame(data=putative_peaks, columns=["putative_peaks_index"])

        for k, all_vals in it_data_indexes.items():
            valid_ind = ~np.isnan(all_vals)
            vals = all_vals[valid_ind].astype(int)
            putative_peaks_df[k + "_index"] = np.nan
            putative_peaks_df[k + "_t"] = np.nan
            putative_peaks_df[k + "_i"] = np.nan

            if len(vals) > 0:
                putative_peaks_df.loc[valid_ind, k + "_index"] = vals
                putative_peaks_df.loc[valid_ind, k + "_t"] = t[vals]
                putative_peaks_df.loc[valid_ind, k + "_i"] = i[vals]

            #if i is not None:
            #    spikes_df[k + "_i"] = np.nan
            #    if len(vals) > 0:
            #        spikes_df.loc[valid_ind, k + "_i"] = i[vals]


        return putative_peaks_df

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
                #burst_tol=0.5, pause_cost=1.0,
                 #deflect_type="min",
                 deflect_type=None,
                 stim_amp_fn=None,
                 baseline_interval=0.1, filter_frequency=1.0,
                 peak_width=0.005):
        self.start = start
        self.end = end
        #self.burst_tol = burst_tol
        #self.pause_cost = pause_cost
        #self.pause_cost_weight = pause_cost_weight
        self.deflect_type = deflect_type
        self.stim_amp_fn = stim_amp_fn
        self.baseline_interval = baseline_interval
        self.filter_frequency = filter_frequency
        self.peak_width = peak_width

    def process(self, t, i, peaks_df, extra_features=None, exclude_clipped=False):
        features = strf.basic_cellattached_train_features(t, peaks_df, self.start, self.end, exclude_clipped=exclude_clipped)

        if self.start is None:
            self.start = 0.0

        if extra_features is None:
            extra_features = []

        #if 'i_baseline' in extra_features:
        #    features['v_baseline'] = subf.baseline_voltage(t, v, self.start, self.baseline_interval, self.filter_frequency)

        #if features["avg_rate"] > 0:
            # We need trough details for these:
            #if 'pause' in extra_features:
            #    features['pause'] = strf.pause(t, peaks_df, self.start, self.end, self.pause_cost_weight)
            #if 'burst' in extra_features:
            #    features['burst'] = strf.burst(t, peaks_df, self.burst_tol, self.pause_cost)
            
        return features




