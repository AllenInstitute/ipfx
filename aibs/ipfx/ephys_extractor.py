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

class EphysSweepFeatureExtractor(object):
    """Feature calculation for a sweep (voltage and/or current time series)."""

    def __init__(self, t=None, v=None, i=None, start=None, end=None, filter=10.,
                 dv_cutoff=20., max_interval=0.005, min_height=2., min_peak=-30.,
                 thresh_frac=0.05, baseline_interval=0.1, baseline_detect_thresh=0.3,
                 id=None):
        """Initialize SweepFeatures object.

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
        baseline_interval: interval length for baseline voltage calculation (before start if start is defined, default 0.1)
        baseline_detect_thresh : dV/dt threshold for evaluating flatness of baseline region (optional, default 0.3)
        """
        self.id = id
        self.t = t
        self.v = v
        self.i = i
        self.start = start
        self.end = end
        self.filter = filter
        self.dv_cutoff = dv_cutoff
        self.max_interval = max_interval
        self.min_height = min_height
        self.min_peak = min_peak
        self.thresh_frac = thresh_frac
        self.baseline_interval = baseline_interval
        self.baseline_detect_thresh = baseline_detect_thresh
        self.stimulus_amplitude_calculator = None

        self._sweep_features = {}
        self._affected_by_clipping = []

    def process_spikes(self):
        """Perform spike-related feature analysis"""
        self._process_individual_spikes()
        self._process_spike_related_features()

    def _process_individual_spikes(self):
        v = self.v
        t = self.t
        dvdt = ft.calculate_dvdt(v, t, self.filter)

        # Basic features of spikes
        putative_spikes = ft.detect_putative_spikes(v, t, self.start, self.end,
                                                    self.filter, self.dv_cutoff)
        peaks = ft.find_peak_indexes(v, t, putative_spikes, self.end)
        putative_spikes, peaks = ft.filter_putative_spikes(v, t, putative_spikes, peaks,
                                                           self.min_height, self.min_peak)

        if not putative_spikes.size:
            # Save time if no spikes detected
            self._spikes_df = DataFrame()
            return

        upstrokes = ft.find_upstroke_indexes(v, t, putative_spikes, peaks, self.filter, dvdt)
        thresholds = ft.refine_threshold_indexes(v, t, upstrokes, self.thresh_frac,
                                                 self.filter, dvdt)
        thresholds, peaks, upstrokes, clipped = ft.check_thresholds_and_peaks(v, t, thresholds, peaks,
                                                                     upstrokes, self.end, self.max_interval)
        if not thresholds.size:
            # Save time if no spikes detected
            self._spikes_df = DataFrame()
            return

        # Spike list and thresholds have been refined - now find other features
        upstrokes = ft.find_upstroke_indexes(v, t, thresholds, peaks, self.filter, dvdt)
        troughs = ft.find_trough_indexes(v, t, thresholds, peaks, clipped, self.end)
        downstrokes = ft.find_downstroke_indexes(v, t, peaks, troughs, clipped, self.filter, dvdt)
        trough_details, clipped = ft.analyze_trough_details(v, t, thresholds, peaks, clipped, self.end,
                                                            self.filter, dvdt=dvdt)
        widths = ft.find_widths(v, t, thresholds, peaks, trough_details[1], clipped)

        base_clipped_list = []

        # Points where we care about t, v, and i if available
        vit_data_indexes = {
            "threshold": thresholds,
            "peak": peaks,
            "trough": troughs,
        }
        base_clipped_list += ["trough"]

        # Points where we care about t and dv/dt
        dvdt_data_indexes = {
            "upstroke": upstrokes,
            "downstroke": downstrokes
        }
        base_clipped_list += ["downstroke"]

        # Trough details
        isi_types = trough_details[0]
        trough_detail_indexes = dict(zip(["fast_trough", "adp", "slow_trough"], trough_details[1:]))
        base_clipped_list += ["fast_trough", "adp", "slow_trough"]

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

            if self.i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.ix[valid_ind, k + "_i"] = self.i[vals]

            if k in base_clipped_list:
                self._affected_by_clipping += [
                    k + "_index",
                    k + "_t",
                    k + "_v",
                    k + "_i",
                ]

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

                if k in base_clipped_list:
                    self._affected_by_clipping += [
                    k + "_index",
                    k + "_t",
                    k + "_v",
                    k,
                ]

        spikes_df["isi_type"] = isi_types
        self._affected_by_clipping += ["isi_type"]

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

            if self.i is not None:
                spikes_df[k + "_i"] = np.nan
                if len(vals) > 0:
                    spikes_df.ix[valid_ind, k + "_i"] = self.i[vals]

            if k in base_clipped_list:
                self._affected_by_clipping += [
                    k + "_index",
                    k + "_t",
                    k + "_v",
                    k + "_i",
                ]

        spikes_df["width"] = widths
        self._affected_by_clipping += ["width"]

        spikes_df["upstroke_downstroke_ratio"] = spikes_df["upstroke"] / -spikes_df["downstroke"]
        self._affected_by_clipping += ["upstroke_downstroke_ratio"]

        self._spikes_df = spikes_df

    def _process_spike_related_features(self):
        t = self.t

        if len(self._spikes_df) == 0:
            self._sweep_features["avg_rate"] = 0
            return

        thresholds = self._spikes_df["threshold_index"].values.astype(int)
        isis = ft.get_isis(t, thresholds)
        with warnings.catch_warnings():
            # ignore mean of empty slice warnings here
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

            sweep_level_features = {
                "adapt": ft.adaptation_index(isis),
                "latency": ft.latency(t, thresholds, self.start),
                "isi_cv": (isis.std() / isis.mean()) if len(isis) >= 1 else np.nan,
                "mean_isi": isis.mean() if len(isis) > 0 else np.nan,
                "median_isi": np.median(isis),
                "first_isi": isis[0] if len(isis) >= 1 else np.nan,
                "avg_rate": ft.average_rate(t, thresholds, self.start, self.end),
            }

        for k, v in six.iteritems(sweep_level_features):
            self._sweep_features[k] = v

    def _process_pauses(self, cost_weight=1.0):
        # Pauses are unusually long ISIs with a "detour reset" among delay resets
        thresholds = self._spikes_df["threshold_index"].values.astype(int)
        isis = ft.get_isis(self.t, thresholds)
        isi_types = self._spikes_df["isi_type"][:-1].values

        return ft.detect_pauses(isis, isi_types, cost_weight)

    def pause_metrics(self):
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

        thresholds = self._spikes_df["threshold_index"].values.astype(int)
        isis = ft.get_isis(self.t, thresholds)

        weight = 1.0
        pause_list = self._process_pauses(weight)

        if len(pause_list) == 0:
            return 0, 0.

        n_pauses = len(pause_list)
        pause_frac = isis[pause_list].sum()
        pause_frac /= self.end - self.start

        return n_pauses, pause_frac

    def _process_bursts(self, tol=0.5, pause_cost=1.0):
        thresholds = self._spikes_df["threshold_index"].values.astype(int)
        isis = ft.get_isis(self.t, thresholds)

        isi_types = self._spikes_df["isi_type"][:-1].values

        fast_tr_v = self._spikes_df["fast_trough_v"].values
        fast_tr_t = self._spikes_df["fast_trough_t"].values
        slow_tr_v = self._spikes_df["slow_trough_v"].values
        slow_tr_t = self._spikes_df["slow_trough_t"].values
        thr_v = self._spikes_df["threshold_v"].values

        bursts = ft.detect_bursts(isis, isi_types, fast_tr_v, fast_tr_t, slow_tr_v, slow_tr_t,
                  thr_v, tol, pause_cost)

        return np.array(bursts)

    def burst_metrics(self):
        """Find bursts and return max "burstiness" index (normalized max rate in burst vs out).

        Returns
        -------
        max_burstiness_index : max "burstiness" index across detected bursts
        num_bursts : number of bursts detected
        """

        burst_info = self._process_bursts()

        if burst_info.shape[0] > 0:
            return burst_info[:, 0].max(), burst_info.shape[0]
        else:
            return 0., 0

    def delay_metrics(self):
        """Calculates ratio of latency to dominant time constant of rise before spike

        Returns
        -------
        delay_ratio : ratio of latency to tau (higher means more delay)
        tau : dominant time constant of rise before spike
        """

        if len(self._spikes_df) == 0:
            logging.info("No spikes available for delay calculation")
            return 0., 0.
        start = self.start
        spike_time = self._spikes_df["threshold_t"].values[0]

        tau = ft.fit_prespike_time_constant(self.v, self.t, start, spike_time)
        latency = spike_time - start

        delay_ratio = latency / tau
        return delay_ratio, tau

    def _get_baseline_voltage(self):
        v = self.v
        t = self.t
        filter_frequency = 1. # in kHz

        # Look at baseline interval before start if start is defined
        if self.start is not None:
            return ft.average_voltage(v, t, self.start - self.baseline_interval, self.start)

        # Otherwise try to find an interval where things are pretty flat
        dv = ft.calculate_dvdt(v, t, filter_frequency)
        non_flat_points = np.flatnonzero(np.abs(dv >= self.baseline_detect_thresh))
        flat_intervals = t[non_flat_points[1:]] - t[non_flat_points[:-1]]
        long_flat_intervals = np.flatnonzero(flat_intervals >= self.baseline_interval)
        if long_flat_intervals.size > 0:
            interval_index = long_flat_intervals[0] + 1
            baseline_end_time = t[non_flat_points[interval_index]]
            return ft.average_voltage(v, t, baseline_end_time - self.baseline_interval,
                                      baseline_end_time)
        else:
            logging.info("Could not find sufficiently flat interval for automatic baseline voltage", RuntimeWarning)
            return np.nan

    def voltage_deflection(self, deflect_type=None):
        """Measure deflection (min or max, between start and end if specified).

        Parameters
        ----------
        deflect_type : measure minimal ('min') or maximal ('max') voltage deflection
            If not specified, it will check to see if the current (i) is positive or negative
            between start and end, then choose 'max' or 'min', respectively
            If the current is not defined, it will default to 'min'.

        Returns
        -------
        deflect_v : peak
        deflect_index : index of peak deflection
        """

        deflect_dispatch = {
            "min": np.argmin,
            "max": np.argmax,
        }

        start = self.start
        if not start:
            start = 0
        start_index = ft.find_time_index(self.t, start)

        end = self.end
        if not end:
            end = self.t[-1]
        end_index = ft.find_time_index(self.t, end)


        if deflect_type is None:
            if self.i is not None:
                halfway_index = ft.find_time_index(self.t, (end - start) / 2. + start)
                if self.i[halfway_index] >= 0:
                    deflect_type = "max"
                else:
                    deflect_type = "min"
            else:
                deflect_type = "min"

        deflect_func = deflect_dispatch[deflect_type]

        v_window = self.v[start_index:end_index]
        deflect_index = deflect_func(v_window) + start_index

        return self.v[deflect_index], deflect_index

    def stimulus_amplitude(self):
        """ """
        if self.stimulus_amplitude_calculator is not None:
            return self.stimulus_amplitude_calculator(self)
        else:
            return np.nan

    def estimate_time_constant(self):
        """Calculate the membrane time constant by fitting the voltage response with a
        single exponential.

        Returns
        -------
        tau : membrane time constant in seconds
        """

        # Assumes this is being done on a hyperpolarizing step
        v_peak, peak_index = self.voltage_deflection("min")
        v_baseline = self.sweep_feature("v_baseline")

        if self.start:
            start_index = ft.find_time_index(self.t, self.start)
        else:
            start_index = 0

        frac = 0.1
        search_result = np.flatnonzero(self.v[start_index:] <= frac * (v_peak - v_baseline) + v_baseline)
        if not search_result.size:
            raise ft.FeatureError("could not find interval for time constant estimate")
        fit_start = self.t[search_result[0] + start_index]
        fit_end = self.t[peak_index]

        a, inv_tau, y0 = ft.fit_membrane_time_constant(self.v, self.t, fit_start, fit_end)

        return 1. / inv_tau

    def estimate_sag(self, peak_width=0.005):
        """Calculate the sag in a hyperpolarizing voltage response.

        Parameters
        ----------
        peak_width : window width to get more robust peak estimate in sec (default 0.005)

        Returns
        -------
        sag : fraction that membrane potential relaxes back to baseline
        """

        t = self.t
        v = self.v

        start = self.start
        if not start:
            start = 0

        end = self.end
        if not end:
            end = self.t[-1]

        v_peak, peak_index = self.voltage_deflection("min")
        v_peak_avg = ft.average_voltage(v, t, start=t[peak_index] - peak_width / 2.,
                                      end=t[peak_index] + peak_width / 2.)
        v_baseline = self.sweep_feature("v_baseline")
        v_steady = ft.average_voltage(v, t, start=end - self.baseline_interval, end=end)
        sag = (v_peak_avg - v_steady) / (v_peak_avg - v_baseline)
        return sag

    def spikes(self):
        """Get all features for each spike as a list of records."""
        return self._spikes_df.to_dict('records')

    def spike_feature(self, key, include_clipped=False, force_exclude_clipped=False):
        """Get specified feature for every spike.

        Parameters
        ----------
        key : feature name
        include_clipped: return values for every identified spike, even when clipping means they will be incorrect/undefined

        Returns
        -------
        spike_feature_values : ndarray of features for each spike
        """

        if not hasattr(self, "_spikes_df"):
            raise AttributeError("EphysSweepFeatureExtractor instance attribute with spike information does not exist yet - have spikes been processed?")

        if len(self._spikes_df) == 0:
            return np.array([])

        if key not in self._spikes_df.columns:
            raise KeyError("requested feature '{:s}' not available".format(key))

        values = self._spikes_df[key].values

        if include_clipped and force_exclude_clipped:
            raise ValueError("include_clipped and force_exclude_clipped cannot both be true")

        if not include_clipped and self.is_spike_feature_affected_by_clipping(key):
            values = values[~self._spikes_df["clipped"].values]
        elif force_exclude_clipped:
            values = values[~self._spikes_df["clipped"].values]

        return values

    def is_spike_feature_affected_by_clipping(self, key):
        return key in self._affected_by_clipping

    def spike_feature_keys(self):
        """Get list of every available spike feature."""
        return self._spikes_df.columns.values.tolist()

    def sweep_feature(self, key, allow_missing=False):
        """Get sweep-level feature (`key`).

        Parameters
        ----------
        key : name of sweep-level feature
        allow_missing : return np.nan if key is missing for sweep (default False)

        Returns
        -------
        sweep_feature : sweep-level feature value
        """

        on_request_dispatch = {
            "v_baseline": self._get_baseline_voltage,
            "tau": self.estimate_time_constant,
            "sag": self.estimate_sag,
            "peak_deflect": self.voltage_deflection,
            "stim_amp": self.stimulus_amplitude,
        }

        if allow_missing and key not in self._sweep_features and key not in on_request_dispatch:
            return np.nan
        elif key not in self._sweep_features and key not in on_request_dispatch:
            raise KeyError("requested feature '{:s}' not available".format(key))

        if key not in self._sweep_features and key in on_request_dispatch:
            fn = on_request_dispatch[key]
            if fn is not None:
                self._sweep_features[key] = fn()
            else:
                raise KeyError("requested feature '{:s}' not defined".format(key))

        return self._sweep_features[key]

    def process_new_spike_feature(self, feature_name, feature_func, affected_by_clipping=False):
        """Add new spike-level feature calculation function

           The function should take this sweep extractor as its argument. Its results
           can be accessed by calling the method spike_feature(<feature_name>).
        """

        if feature_name in self._spikes_df.columns:
            raise KeyError("Feature {:s} already exists for sweep".format(feature_name))

        self._spikes_df[feature_name] = feature_func(self)

        if affected_by_clipping:
            self._affected_by_clipping.append(feature_name)

    def process_new_sweep_feature(self, feature_name, feature_func):
        """Add new sweep-level feature calculation function

           The function should take this sweep extractor as its argument. Its results
           can be accessed by calling the method sweep_feature(<feature_name>).
        """

        if feature_name in self._sweep_features:
            raise KeyError("Feature {:s} already exists for sweep".format(feature_name))

        self._sweep_features[feature_name] = feature_func(self)

    def set_stimulus_amplitude_calculator(self, function):
        self.stimulus_amplitude_calculator = function

    def sweep_feature_keys(self):
        """Get list of every available sweep-level feature."""
        return self._sweep_features.keys()

    def as_dict(self):
        """Create dict of features and spikes."""
        output_dict = self._sweep_features.copy()
        output_dict["spikes"] = self.spikes()
        if self.id is not None:
            output_dict["id"] = self.id
        return output_dict


class EphysSweepSetFeatureExtractor(object):
    def __init__(self, t_set=None, v_set=None, i_set=None, start=None, end=None,
                 filter=10., dv_cutoff=20., max_interval=0.005, min_height=2.,
                 min_peak=-30., thresh_frac=0.05, baseline_interval=0.1,
                 baseline_detect_thresh=0.3, id_set=None):
        """Initialize EphysSweepSetFeatureExtractor object.

        Parameters
        ----------
        t_set : list of ndarray of times in seconds
        v_set : list of ndarray of voltages in mV
        i_set : list of ndarray of currents in pA
        start : start of time window for feature analysis (optional, can be list)
        end : end of time window for feature analysis (optional, can be list)
        filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
        dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
        max_interval : maximum acceptable time between start of spike and time of peak in sec (optional, default 0.005)
        min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
        min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
        thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
        baseline_interval: interval length for baseline voltage calculation (before start if start is defined, default 0.1)
        baseline_detect_thresh : dV/dt threshold for evaluating flatness of baseline region (optional, default 0.3)
        """

        if t_set is not None and v_set is not None:
            self._set_sweeps(t_set, v_set, i_set, start, end, filter, dv_cutoff, max_interval,
                             min_height, min_peak, thresh_frac, baseline_interval,
                             baseline_detect_thresh, id_set)
        else:
            self._sweeps = None

    @classmethod
    def from_sweeps(cls, sweep_list):
        """Initialize EphysSweepSetFeatureExtractor object with a list of pre-existing
        sweep feature extractor objects.
        """

        obj = cls()
        obj._sweeps = sweep_list
        return obj

    def _set_sweeps(self, t_set, v_set, i_set, start, end, filter, dv_cutoff, max_interval,
                    min_height, min_peak, thresh_frac, baseline_interval,
                    baseline_detect_thresh, id_set):
        if type(t_set) != list:
            raise ValueError("t_set must be a list")

        if type(v_set) != list:
            raise ValueError("v_set must be a list")

        if i_set is not None and type(i_set) != list:
            raise ValueError("i_set must be a list")

        if len(t_set) != len(v_set):
            raise ValueError("t_set and v_set must have the same number of items")

        if i_set and len(t_set) != len(i_set):
            raise ValueError("t_set and i_set must have the same number of items")

        if id_set is None:
            id_set = range(len(t_set))
        if len(id_set) != len(t_set):
            raise ValueError("t_set and id_set must have the same number of items")

        sweeps = []
        if i_set is None:
            i_set = [None] * len(t_set)

        if type(start) is not list:
            start = [start] * len(t_set)
            end = [end] * len(t_set)

        sweeps = [ EphysSweepFeatureExtractor(t, v, i, start, end,
                                              filter=filter, dv_cutoff=dv_cutoff,
                                              max_interval=max_interval,
                                              min_height=min_height, min_peak=min_peak,
                                              thresh_frac=thresh_frac,
                                              baseline_interval=baseline_interval,
                                              baseline_detect_thresh=baseline_detect_thresh,
                                              id=sid) \
                       for t, v, i, start, end, sid in zip(t_set, v_set, i_set, start, end, id_set) ]

        self._sweeps = sweeps

    def sweeps(self):
        """Get list of EphysSweepFeatureExtractor objects."""
        return self._sweeps

    def process_spikes(self):
        """Analyze spike features for all sweeps."""
        for sweep in self._sweeps:
            sweep.process_spikes()

    def sweep_features(self, key, allow_missing=False):
        """Get nparray of sweep-level feature (`key`) for all sweeps

        Parameters
        ----------
        key : name of sweep-level feature
        allow_missing : return np.nan if key is missing for sweep (default False)

        Returns
        -------
        sweep_feature : nparray of sweep-level feature values
        """

        return np.array([swp.sweep_feature(key, allow_missing) for swp in self._sweeps])

    def spike_feature_averages(self, key):
        """Get nparray of average spike-level feature (`key`) for all sweeps"""
        return np.array([swp.spike_feature(key).mean() for swp in self._sweeps])


def input_resistance(ext):
    """Estimate input resistance in MOhms, assuming all sweeps in passed extractor
    are hyperpolarizing responses."""

    sweeps = ext.sweeps()
    if not sweeps:
        raise ft.FeatureError("no sweeps available for input resistance calculation")

    v_vals = []
    i_vals = []
    for sweep in sweeps:
        if sweep.i is None:
            raise ft.FeatureError("cannot calculate input resistance: i not defined for a sweep")

        v_peak, min_index = sweep.voltage_deflection('min')
        v_vals.append(v_peak)
        i_vals.append(sweep.i[min_index])

    v = np.array(v_vals)
    i = np.array(i_vals)

    if len(v) == 1:
        # If there's just one sweep, we'll have to use its own baseline to estimate
        # the input resistance
        v = np.append(v, sweeps[0].sweep_feature("v_baseline"))
        i = np.append(i, 0.)

    A = np.vstack([i, np.ones_like(i)]).T
    m, c = np.linalg.lstsq(A, v)[0]

    return m * 1e3


def membrane_time_constant(ext):
    """Average the membrane time constant values estimated from each sweep in passed extractor."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        avg_tau = np.nanmean(ext.sweep_features("tau"))
    return avg_tau


def fit_fi_slope(ext):
    """Fit the rate and stimulus amplitude to a line and return the slope of the fit."""
    if len(ext.sweeps()) < 2:
        raise ft.FeatureError("Cannot fit f-I curve slope with less than two suprathreshold sweeps")

    x = np.array(map(_step_stim_amp, ext.sweeps()))
    y = ext.sweep_features("avg_rate")

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y)[0]

    return m


#def reset_long_squares_start(when):
#    global LONG_SQUARES_START, LONG_SQUARES_END
#    delta = LONG_SQUARES_END - LONG_SQUARES_START
#    LONG_SQUARES_START = when
#    LONG_SQUARES_END = when + delta




def _step_stim_amp(sweep):
    t_index = ft.find_time_index(sweep.t, sweep.start)
    return sweep.i[t_index + 1]


def _short_step_stim_amp(sweep):
    t_index = ft.find_time_index(sweep.t, sweep.start)
    return sweep.i[t_index + 1:].max()
