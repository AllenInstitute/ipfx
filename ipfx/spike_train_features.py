import numpy as np
import warnings
import logging
from . import spike_features as spkf
from . import error as er


def basic_spike_train_features(t, spikes_df, start, end, exclude_clipped=False):
    features = {}
    if len(spikes_df) == 0 or spikes_df.empty:
        features["avg_rate"] = 0
        return features

    thresholds = spikes_df["threshold_index"].values.astype(int)
    if exclude_clipped:
        mask = spikes_df["clipped"].values.astype(bool)
        thresholds = thresholds[~mask]
    isis = get_isis(t, thresholds)
    with warnings.catch_warnings():
        # ignore mean of empty slice warnings here
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

        features = {
            "adapt": adaptation_index(isis),
            "latency": latency(t, thresholds, start),
            "isi_cv": (isis.std() / isis.mean()) if len(isis) >= 1 else np.nan,
            "mean_isi": isis.mean() if len(isis) > 0 else np.nan,
            "median_isi": np.median(isis),
            "first_isi": isis[0] if len(isis) >= 1 else np.nan,
            "avg_rate": average_rate(t, thresholds, start, end),
        }

    return features


def pause(t, spikes_df, start, end, cost_weight=1.0):
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
    isis = get_isis(t, thresholds)
    isi_types = spikes_df["isi_type"][:-1].values

    pause_list = spkf.detect_pauses(isis, isi_types, cost_weight)

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
    isis = get_isis(t, thresholds)

    isi_types = spikes_df["isi_type"][:-1].values
    fast_tr_v = spikes_df["fast_trough_v"].values
    fast_tr_t = spikes_df["fast_trough_t"].values
    slow_tr_v = spikes_df["slow_trough_v"].values
    slow_tr_t = spikes_df["slow_trough_t"].values
    thr_v = spikes_df["threshold_v"].values

    bursts = spkf.detect_bursts(isis, isi_types,
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

    tau = spkf.fit_prespike_time_constant(t, v, start, spike_time)
    latency = spike_time - start

    delay_ratio = latency / tau
    return delay_ratio, tau


def fit_fi_slope(stim_amps, avg_rates):
    """Fit the rate and stimulus amplitude to a line and return the slope of the fit."""

    if len(stim_amps) < 2:
        raise er.FeatureError("Cannot fit f-I curve slope with less than two sweeps")

    x = stim_amps
    y = avg_rates

    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y,rcond=None)[0]

    return m


def get_isis(t, spikes):
    """Find interspike intervals in sec between spikes (as indexes)."""

    if len(spikes) <= 1:
        return np.array([])

    return t[spikes[1:]] - t[spikes[:-1]]


def adaptation_index(isis):
    """Calculate adaptation index of `isis`."""
    if len(isis) == 0:
        return np.nan

    return norm_diff(isis)


def latency(t, spikes, start):
    """Calculate time to the first spike."""

    if len(spikes) == 0:
        return np.nan

    if start is None:
        start = t[0]

    return t[spikes[0]] - start


def average_rate(t, spikes, start, end):
    """Calculate average firing rate during interval between `start` and `end`.

    Parameters
    ----------
    t : numpy array of times in seconds
    spikes : numpy array of spike indexes
    start : start of time window for spike detection
    end : end of time window for spike detection

    Returns
    -------
    avg_rate : average firing rate in spikes/sec
    """

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    spikes_in_interval = [spk for spk in spikes if t[spk] >= start and t[spk] <= end]
    avg_rate = len(spikes_in_interval) / (end - start)
    return avg_rate


def norm_diff(a):
    """Calculate average of (a[i] - a[i+1]) / (a[i] + a[i+1])."""

    if len(a) <= 1:
        return np.nan

    a = a.astype(float)
    if np.allclose((a[1:] + a[:-1]), 0.):
        return 0.
    norm_diffs = (a[1:] - a[:-1]) / (a[1:] + a[:-1])
    norm_diffs[(a[1:] == 0) & (a[:-1] == 0)] = 0.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
        avg = np.nanmean(norm_diffs)
    return avg


def norm_sq_diff(a):
    """Calculate average of (a[i] - a[i+1])^2 / (a[i] + a[i+1])^2."""
    if len(a) <= 1:
        return np.nan

    a = a.astype(float)
    norm_sq_diffs = np.square((a[1:] - a[:-1])) / np.square((a[1:] + a[:-1]))
    return norm_sq_diffs.mean()


def detect_pauses(isis, isi_types, cost_weight=1.0):
    """Determine which ISIs are "pauses" in ongoing firing.

    Pauses are unusually long ISIs with a "detour reset" among "direct resets".

    Parameters
    ----------
    isis : numpy array of interspike intervals
    isi_types : numpy array of interspike interval types ('direct' or 'detour')
    cost_weight : weight for cost function for calling an ISI a pause
        Higher cost weights lead to fewer ISIs identified as pauses. The cost function
        also depends on the difference between the duration of the "pause" ISIs and the
        average duration and standard deviation of "non-pause" ISIs.

    Returns
    -------
    pauses : numpy array of indices corresponding to pauses in `isis`
    """

    if len(isis) != len(isi_types):
        raise er.FeatureError("Wrong number of ISIs")

    if not np.any(isi_types == "direct"):
        # Need some direct-type firing to have pauses
        return np.array([])

    detour_candidates = [i for i, isi_type in enumerate(isi_types) if isi_type == "detour"]
    median_direct = np.median(isis[isi_types == "direct"])
    direct_candidates = [i for i, isi_type in enumerate(isi_types) if isi_type == "direct" and isis[i] > 3 * median_direct]
    candidates = detour_candidates + direct_candidates

    if not candidates:
        return np.array([])

    pause_list = np.array([], dtype=int)
    all_cv = isis.std() / isis.mean()
    best_net = 0
    for i in candidates:
        temp_pause_list = np.append(pause_list, i)
        non_pause_isis = np.delete(isis, temp_pause_list)
        pause_isis = isis[temp_pause_list]
        if len(non_pause_isis) < 2:
            break
        cv = non_pause_isis.std() / non_pause_isis.mean()
        benefit = all_cv - cv
        cost = np.sum(non_pause_isis.std() / np.abs(non_pause_isis.mean() - pause_isis))
        cost *= cost_weight
        net = benefit - cost
        if net > 0 and net < best_net:
            break
        if net > best_net:
            best_net = net
        pause_list = np.append(pause_list, i)

    if best_net <= 0:
        pause_list = np.array([])

    return np.sort(pause_list)


def detect_bursts(isis, isi_types, fast_tr_v, fast_tr_t, slow_tr_v, slow_tr_t,
                  thr_v, tol=0.5, pause_cost=1.0):
    """Detect bursts in spike train.

    Parameters
    ----------
    isis : numpy array of n interspike intervals
    isi_types : numpy array of n interspike interval types
    fast_tr_v : numpy array of fast trough voltages for the n + 1 spikes of the train
    fast_tr_t : numpy array of fast trough times for the n + 1 spikes of the train
    slow_tr_v : numpy array of slow trough voltages for the n + 1 spikes of the train
    slow_tr_t : numpy array of slow trough times for the n + 1 spikes of the train
    thr_v : numpy array of threshold voltages for the n + 1 spikes of the train
    tol : tolerance for the difference in slow trough voltages and thresholds (default 0.5 mV)
        Used to identify "delay" interspike intervals that occur within a burst

    Returns
    -------
    bursts : list of bursts
        Each item in list is a tuple of the form (burst_index, start, end) where `burst_index`
        is a comparison index between the highest instantaneous rate within the burst vs
        the highest instantaneous rate outside the burst. `start` is the index of the first
        ISI of the burst, and `end` is the ISI index immediately following the burst.
    """

    if len(isis) != len(isi_types):
        raise er.FeatureError("Wrong number of ISIs")

    if len(isis) < 2: # can't determine burstiness for a single ISI
        return np.array([])

    fast_tr_v = fast_tr_v[:-1]
    fast_tr_t = fast_tr_t[:-1]
    slow_tr_v = slow_tr_v[:-1]
    slow_tr_t = slow_tr_t[:-1]

    isi_types = np.array(isi_types) # don't want to change the actual isi types data

    # Burst transitions can't be at "pause"-like ISIs
    pauses = detect_pauses(isis, isi_types, cost_weight=pause_cost).astype(int)
    isi_types[pauses] = "pauselike"

    if not (np.any(isi_types == "direct") and np.any(isi_types == "detour")):
        # no candidates that could be bursts
        return np.array([])

    # Want to catch special case of detour in the middle of a large burst where
    # the slow trough value is higher than the previous spike's threshold
    isi_types[(thr_v[:-1] < (slow_tr_v + tol)) & (isi_types == "detour")] = "midburst"

    # Find transitions from direct -> detour and vice versa for burst boundaries
    into_burst = np.array([i + 1 for i, (prev, cur) in
                 enumerate(zip(isi_types[:-1], isi_types[1:])) if
                 cur == "direct" and prev == "detour"],
                 dtype=int)
    if isi_types[0] == "direct":
        into_burst = np.append(np.array([0]), into_burst)

    drop_into = []
    out_of_burst = []
    for j, (into, next) in enumerate(zip(into_burst, np.append(into_burst[1:], len(isis)))):
        for i, isi in enumerate(isi_types[into + 1:next]):
            if isi == "detour":
                out_of_burst.append(i + into + 1)
                break
            elif isi == "pauselike":
                drop_into.append(j)
                break

    mask = np.ones_like(into_burst, dtype=bool)
    mask[drop_into] = False
    into_burst = into_burst[mask]

    out_of_burst = np.array(out_of_burst, dtype=int)
    if len(out_of_burst) == len(into_burst) - 1:
        out_of_burst = np.append(out_of_burst, len(isi_types))

    if not (into_burst.size or out_of_burst.size):
        return np.array([])

    if len(into_burst) != len(out_of_burst):
        raise er.FeatureError("Inconsistent burst boundary identification")

    inout_pairs = list(zip(into_burst, out_of_burst))
    delta_t = slow_tr_t - fast_tr_t

    scores = _score_burst_set(inout_pairs, isis, delta_t)
    best_score = np.mean(scores)
    worst = np.argmin(scores)
    test_bursts = list(inout_pairs)
    del test_bursts[worst]
    while len(test_bursts) > 0:
        scores = _score_burst_set(test_bursts, isis, delta_t)
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            inout_pairs = list(test_bursts)
            worst = np.argmin(scores)
            del test_bursts[worst]
        else:
            break

    if best_score < 0:
        return np.array([])

    bursts = []
    for i, (into, outof) in enumerate(inout_pairs):
        if i == len(inout_pairs) - 1: # last burst to evaluate
            if outof <= len(isis) - 1: # are there spikes left after the burst?
                metric = _burstiness_index(isis[into:outof], isis[outof:])
            elif i == 0: # was this the first one (and there weren't spikes after)?
                metric = _burstiness_index(isis[into:outof], isis[:into])
            else:
                prev_burst = inout_pairs[i - 1]
                metric = _burstiness_index(isis[into:outof], isis[prev_burst[1]:into])
        else:
            next_burst = inout_pairs[i + 1]
            metric = _burstiness_index(isis[into:outof], isis[outof:next_burst[0]])
        bursts.append((metric, into, outof))

    return bursts


def _score_burst_set(bursts, isis, delta_t, c_n=0.1, c_tx=0.01):
    in_burst = np.zeros_like(isis, dtype=bool)
    for b in bursts:
        in_burst[b[0]:b[1]] = True

    # If all ISIs are part of a burst, give it a bad score
    if len(isis[~in_burst]) == 0:
        return [-1e12] * len(bursts)

    delta_frac = delta_t / isis

    scores = []
    for b in bursts:
        score = _burstiness_index(isis[b[0]:b[1]], isis[~in_burst]) # base score
        if b[1] < len(delta_t):
            score -= c_tx * (1. / (delta_frac[b[1]])) # cost for starting a burst
        if b[0] > 0:
            score -= c_tx * (1. / delta_frac[b[0] - 1]) # cost for ending a burst
        score -= c_n * (b[1] - b[0] - 1) # cost for extending a burst
        scores.append(score)

    return scores


def _burstiness_index(in_burst_isis, out_burst_isis):
    burst_rate = 1. / in_burst_isis.min()
    out_rate = 1. / out_burst_isis.min()
    return (burst_rate - out_rate) / (burst_rate + out_rate)
