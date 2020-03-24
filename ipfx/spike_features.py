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
import logging
import numpy as np
from scipy.optimize import curve_fit
from functools import partial
from . import time_series_utils as tsu
from . import spike_detector as spkd
from . import error as er

def find_widths(v, t, spike_indexes, peak_indexes, trough_indexes, clipped=None):
    """Find widths at half-height for spikes.

    Widths are only returned when heights are defined

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of trough indexes

    Returns
    -------
    widths : numpy array of spike widths in sec
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if len(spike_indexes) < len(trough_indexes):
        raise er.FeatureError("Cannot have more troughs than spikes")

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    use_indexes = ~np.isnan(trough_indexes)
    use_indexes[clipped] = False

    heights = np.zeros_like(trough_indexes) * np.nan
    heights[use_indexes] = v[peak_indexes[use_indexes]] - v[trough_indexes[use_indexes].astype(int)]

    width_levels = np.zeros_like(trough_indexes) * np.nan
    width_levels[use_indexes] = heights[use_indexes] / 2. + v[trough_indexes[use_indexes].astype(int)]


    thresh_to_peak_levels = np.zeros_like(trough_indexes) * np.nan
    thresh_to_peak_levels[use_indexes] = (v[peak_indexes[use_indexes]] - v[spike_indexes[use_indexes]]) / 2. + v[spike_indexes[use_indexes]]

    # Some spikes in burst may have deep trough but short height, so can't use same
    # definition for width

    width_levels[width_levels < v[spike_indexes]] = thresh_to_peak_levels[width_levels < v[spike_indexes]]

    width_starts = np.zeros_like(trough_indexes) * np.nan
    width_starts[use_indexes] = np.array([pk - np.flatnonzero(v[pk:spk:-1] <= wl)[0] if
                    np.flatnonzero(v[pk:spk:-1] <= wl).size > 0 else np.nan for pk, spk, wl
                    in zip(peak_indexes[use_indexes], spike_indexes[use_indexes], width_levels[use_indexes])])
    width_ends = np.zeros_like(trough_indexes) * np.nan

    width_ends[use_indexes] = np.array([pk + np.flatnonzero(v[pk:tr] <= wl)[0] if
                    np.flatnonzero(v[pk:tr] <= wl).size > 0 else np.nan for pk, tr, wl
                    in zip(peak_indexes[use_indexes], trough_indexes[use_indexes].astype(int), width_levels[use_indexes])])

    missing_widths = np.isnan(width_starts) | np.isnan(width_ends)
    widths = np.zeros_like(width_starts, dtype=np.float64)
    widths[~missing_widths] = t[width_ends[~missing_widths].astype(int)] - \
                              t[width_starts[~missing_widths].astype(int)]
    if any(missing_widths):
        widths[missing_widths] = np.nan

    return widths


def analyze_trough_details(v, t, spike_indexes, peak_indexes, clipped=None, end=None, filter=10.,
                           heavy_filter=1., term_frac=0.01, adp_thresh=0.5, tol=0.5,
                           flat_interval=0.002, adp_max_delta_t=0.005, adp_max_delta_v=10., dvdt=None):
    """Analyze trough to determine if an ADP exists and whether the reset is a 'detour' or 'direct'

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    heavy_filter : lower cutoff frequency for 4-pole low-pass Bessel filter in kHz (default 1)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    adp_thresh: minimum dV/dt in V/s to exceed to be considered to have an ADP (optional, default 1.5)
    tol : tolerance for evaluating whether Vm drops appreciably further after end of spike (default 1.0 mV)
    flat_interval: if the trace is flat for this duration, stop looking for an ADP (default 0.002 s)
    adp_max_delta_t: max possible ADP delta t (default 0.005 s)
    adp_max_delta_v: max possible ADP delta v (default 10 mV)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    isi_types : numpy array of isi reset types (direct or detour)
    fast_trough_indexes : numpy array of indexes at the start of the trough (i.e. end of the spike)
    adp_indexes : numpy array of adp indexes (np.nan if there was no ADP in that ISI
    slow_trough_indexes : numpy array of indexes at the minimum of the slow phase of the trough
                          (if there wasn't just a fast phase)
    """

    if end is None:
        end = t[-1]
    end_index = tsu.find_time_index(t, end)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes)

    # Can't evaluate for spikes that are clipped by the window
    orig_len = len(peak_indexes)
    valid_spike_indexes = spike_indexes[~clipped]
    valid_peak_indexes = peak_indexes[~clipped]

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    dvdt_hvy = tsu.calculate_dvdt(v, t, heavy_filter)

    # Writing as for loop - see if I can vectorize any later
    fast_trough_indexes = []
    adp_indexes = []
    slow_trough_indexes = []
    isi_types = []

    update_clipped = []
    for peak, next_spk in zip(valid_peak_indexes, np.append(valid_spike_indexes[1:], end_index)):
        downstroke = dvdt[peak:next_spk].argmin() + peak
        target = term_frac * dvdt[downstroke]

        terminated_points = np.flatnonzero(dvdt[downstroke:next_spk] >= target)
        if terminated_points.size:
            terminated = terminated_points[0] + downstroke
            update_clipped.append(False)
        else:
            logging.debug("Could not identify fast trough - marking spike as clipped")
            isi_types.append(np.nan)
            fast_trough_indexes.append(np.nan)
            adp_indexes.append(np.nan)
            slow_trough_indexes.append(np.nan)
            update_clipped.append(True)
            continue

        # Could there be an ADP?
        adp_index = np.nan
        dv_over_thresh = np.flatnonzero(dvdt_hvy[terminated:next_spk] >= adp_thresh)
        if dv_over_thresh.size:
            cross = dv_over_thresh[0] + terminated

            # only want to look for ADP before things get pretty flat
            # otherwise, could just pick up random transients long after the spike
            if t[cross] - t[terminated] < flat_interval:
                # Going back up fast, but could just be going into another spike
                # so need to check for a reversal (zero-crossing) in dV/dt
                zero_return_vals = np.flatnonzero(dvdt_hvy[cross:next_spk] <= 0)
                if zero_return_vals.size:
                    putative_adp_index = zero_return_vals[0] + cross
                    min_index = v[putative_adp_index:next_spk].argmin() + putative_adp_index
                    if (v[putative_adp_index] - v[min_index] >= tol and
                            v[putative_adp_index] - v[terminated] <= adp_max_delta_v and
                            t[putative_adp_index] - t[terminated] <= adp_max_delta_t):
                        adp_index = putative_adp_index
                        slow_phase_min_index = min_index
                        isi_type = "detour"

        if np.isnan(adp_index):
            v_term = v[terminated]
            min_index = v[terminated:next_spk].argmin() + terminated
            if v_term - v[min_index] >= tol:
                # dropped further after end of spike -> detour reset
                isi_type = "detour"
                slow_phase_min_index = min_index
            else:
                isi_type = "direct"

        isi_types.append(isi_type)
        fast_trough_indexes.append(terminated)
        adp_indexes.append(adp_index)
        if isi_type == "detour":
            slow_trough_indexes.append(slow_phase_min_index)
        else:
            slow_trough_indexes.append(np.nan)

    # If we had to kick some spikes out before, need to add nans at the end
    output = []
    output.append(np.array(isi_types))
    for d in (fast_trough_indexes, adp_indexes, slow_trough_indexes):
        output.append(np.array(d, dtype=float))

    if orig_len > len(isi_types):
        extra = np.zeros(orig_len - len(isi_types)) * np.nan
        output = tuple((np.append(o, extra) for o in output))

    # The ADP and slow trough for the last spike in a train are not reliably
    # calculated, and usually extreme when wrong, so we will NaN them out.
    #
    # Note that this will result in a 0 value when delta V or delta T is
    # calculated, which may not be strictly accurate to the trace, but the
    # magnitude of the difference will be less than in many of the erroneous
    # cases seen otherwise

    output[2][-1] = np.nan # ADP
    output[3][-1] = np.nan # slow trough

    clipped[~clipped] = update_clipped
    return output, clipped


def fit_prespike_time_constant(t, v, start, spike_time, dv_limit=-0.001, tau_limit=0.3):
    """Finds the dominant time constant of the pre-spike rise in voltage

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of voltage rise (seconds)
    spike_time : time of first spike (seconds)
    dv_limit : dV/dt cutoff (default -0.001)
        Shortens fit window if rate of voltage drop exceeds this limit
    tau_limit : upper bound for slow time constant (seconds, default 0.3)
        If the slower time constant of a double-exponential fit is twice that of the faster
        and exceeds this limit, the faster one will be considered the dominant one

    Returns
    -------
    tau : dominant time constant (seconds)
    """

    start_index = tsu.find_time_index(t, start)
    end_index = tsu.find_time_index(t, spike_time)
    if end_index <= start_index:
        raise er.FeatureError("Start for pre-spike time constant fit cannot be after the spike time.")

    v_slice = v[start_index:end_index]
    t_slice = t[start_index:end_index]

    # Solve linear version with single exponential first to guess at the time constant
    y0 = v_slice.max() + 5e-6 # set y0 slightly above v_slice maximum
    y = -v_slice + y0
    y = np.log(y)

    dy = tsu.calculate_dvdt(y, t_slice, filter=1.0)

    # End the fit interval if the voltage starts dropping
    new_end_indexes = np.flatnonzero(dy <= dv_limit)
    cross_limit = 0.0005 # sec
    if not new_end_indexes.size or t_slice[new_end_indexes[0]] - t_slice[0] < cross_limit:
        # either never crosses or crosses too early
        new_end_index = len(v_slice)
    else:
        new_end_index = new_end_indexes[0]

    K, A_log = np.polyfit(t_slice[:new_end_index] - t_slice[0], y[:new_end_index], 1)
    A = np.exp(A_log)

    dbl_exp_y0 = partial(_dbl_exp_fit, y0)
    try:
        popt, pcov = curve_fit(dbl_exp_y0, t_slice - t_slice[0], v_slice, p0=(-A / 2.0, -1.0 / K, -A / 2.0, -1.0 / K))
    except RuntimeError:
        # Fall back to single fit
        tau = -1.0 / K
        return tau

    # Find dominant time constant
    if popt[1] < popt[3]:
        faster_weight, faster_tau, slower_weight, slower_tau = popt
    else:
        slower_weight, slower_tau, faster_weight, faster_tau = popt

    # These are all empirical values
    if np.abs(faster_weight) > np.abs(slower_weight):
        tau = faster_tau
    elif (slower_tau - faster_tau) / slower_tau <= 0.1: # close enough; just use slower
        tau = slower_tau
    elif slower_tau > tau_limit and slower_weight / faster_weight < 2.0:
        tau = faster_tau
    else:
        tau = slower_tau

    return tau


def _dbl_exp_fit(y0, x, A1, tau1, A2, tau2):
    penalty = 0
    if tau1 < 0 or tau2 < 0:
        penalty = 1e6
    return y0 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + penalty


def estimate_adjusted_detection_parameters(v_set, t_set, interval_start, interval_end, filter=10):
    """
    Estimate adjusted values for spike detection by analyzing a period when the voltage
    changes quickly but passively (due to strong current stimulation), which can result
    in spurious spike detection results.

    Parameters
    ----------
    v_set : list of numpy arrays of voltage time series in mV
    t_set : list of numpy arrays of times in seconds
    interval_start : start of analysis interval (sec)
    interval_end : end of analysis interval (sec)

    Returns
    -------
    new_dv_cutoff : adjusted dv/dt cutoff (V/s)
    new_thresh_frac : adjusted fraction of avg upstroke to find threshold
    """

    if type(v_set) is not list:
        v_set = list(v_set)

    if type(t_set) is not list:
        t_set = list(t_set)

    if len(v_set) != len(t_set):
        raise er.FeatureError("t_set and v_set must be lists of equal size")

    if len(v_set) == 0:
        raise er.FeatureError("t_set and v_set are empty")

    start_index = tsu.find_time_index(t_set[0], interval_start)
    end_index = tsu.find_time_index(t_set[0], interval_end)

    maxes = []
    ends = []
    dv_set = []
    for v, t in zip(v_set, t_set):
        dv = tsu.calculate_dvdt(v, t, filter)
        dv_set.append(dv)
        maxes.append(dv[start_index:end_index].max())
        ends.append(dv[end_index])

    maxes = np.array(maxes)
    ends = np.array(ends)

    cutoff_adj_factor = 1.1
    thresh_frac_adj_factor = 1.2

    new_dv_cutoff = np.median(maxes) * cutoff_adj_factor
    min_thresh = np.median(ends) * thresh_frac_adj_factor

    all_upstrokes = np.array([])
    for v, t, dv in zip(v_set, t_set, dv_set):
        putative_spikes = spkd.detect_putative_spikes(v, t, dv_cutoff=new_dv_cutoff, filter=filter)
        peaks = spkd.find_peak_indexes(v, t, putative_spikes)
        putative_spikes, peaks = spkd.filter_putative_spikes(v, t, putative_spikes, peaks, dvdt=dv, filter=filter)
        upstrokes = spkd.find_upstroke_indexes(v, t, putative_spikes, peaks, dvdt=dv)
        if upstrokes.size:
            all_upstrokes = np.append(all_upstrokes, dv[upstrokes])
    new_thresh_frac = min_thresh / all_upstrokes.mean()

    return new_dv_cutoff, new_thresh_frac


