import numpy as np
from scipy.optimize import curve_fit


def measure_blowout(v, idx0):

    return np.mean(v[idx0:])


def measure_electrode_0(curr, hz, t=0.005):

    n_time_steps = int(t * hz)
    # electrode 0 is the average current reading with zero voltage input
    # (ie, the equivalent of resting potential in current-clamp mode)
    if n_time_steps:
        return np.mean(curr[0:n_time_steps])
    else:
        return None


def measure_seal(v, curr, t, post_transient_shift_ms=0.15):
    avg_i, avg_v, slice_t, rel_up_ind, rel_down_ind = average_cell_attached_pulses(
        v * 1e-3, curr * 1e-12, t)
    return 1e-9 * get_r_from_stable_pulse_response_fit(
        avg_v, avg_i, slice_t,
        rel_up_ind, rel_down_ind, post_transient_shift_ms=post_transient_shift_ms)


def measure_input_resistance(v, curr, t, post_transient_shift_ms=1.0):
    avg_i, avg_v, slice_t, rel_up_ind, rel_down_ind = average_whole_cell_pulses(
        v * 1e-3, curr * 1e-12, t)
    r_a = 1e-6 * get_r_from_peak_pulse_response(
        avg_v, avg_i, slice_t,
        rel_up_ind, rel_down_ind)
    r_tot = 1e-6 * get_r_from_stable_pulse_response_fit(
        avg_v, avg_i, slice_t,
        rel_up_ind, rel_down_ind, post_transient_shift_ms=post_transient_shift_ms)
    return r_tot - r_a


def measure_initial_access_resistance(v, curr, t):
    avg_i, avg_v, slice_t, rel_up_ind, rel_down_ind = average_whole_cell_pulses(
        v * 1e-3, curr * 1e-12, t)
    return 1e-6 * get_r_from_peak_pulse_response(
        avg_v, avg_i, slice_t,
        rel_up_ind, rel_down_ind)


def measure_vm(vals):
    if len(vals) < 1:
        return 0, 0
    mean = np.mean(vals)
    rms = np.sqrt(np.mean(np.square(vals-mean)))
    return float(mean), float(rms)


def measure_vm_delta(mean_start, mean_end):

    if mean_end is not None:
        delta = abs(mean_start - mean_end)
        return float(delta)
    else:
        return None


def get_r_from_stable_pulse_response_fit(avg_v, avg_i, t,
        relative_up_ind, relative_down_ind, post_transient_shift_ms):
    if avg_i is None:
        # Not enough pulses to average
        return np.nan

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    post_transient_shift = int(one_ms * post_transient_shift_ms)

    # baseline - take average v and i one ms before start
    end = relative_up_ind - 1
    start = end - one_ms
    avg_v_base = np.mean(avg_v[start:end])
    avg_i_base = np.mean(avg_i[start:end])

    t_window = (t[relative_up_ind + post_transient_shift:relative_down_ind] -
        t[relative_up_ind + post_transient_shift])
    i_window = avg_i[relative_up_ind + post_transient_shift:relative_down_ind] * 1e12
    guess = (
        max(i_window[0] - i_window[-1], 100),
        1e3,
        max(i_window[-1], avg_i_base * 1e12 + 5)
    )

    popt, pcov = curve_fit(
        _exp_curve, t_window, i_window,
        p0=guess, bounds=([0, 0, avg_i_base * 1e12], [np.inf, np.inf, np.inf])
    )
    pred = _exp_curve(t_window, *popt) * 1e-12

    # steady-state - take average v and i one ms before end
    end = relative_down_ind - 1
    start = end - one_ms
    avg_v_steady = np.mean(avg_v[start:end])
    avg_i_steady = np.mean(avg_i[start:end])
    avg_i_steady = np.mean(pred[-one_ms:])

    r = (avg_v_steady - avg_v_base) / (avg_i_steady - avg_i_base)

    return r


def get_r_from_peak_pulse_response(avg_v, avg_i, t,
        relative_up_ind, relative_down_ind):
    if avg_i is None:
        # Not enough pulses to average
        return np.nan

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    # take average v and i one ms before start of pulse
    end = relative_up_ind - 1
    start = end - one_ms
    avg_v_base = np.mean(avg_v[start:end])
    avg_i_base = np.mean(avg_i[start:end])

    # find peak i during the pulse
    start = relative_up_ind
    end = relative_down_ind - 1
    idx = start + np.argmax(avg_i[start:end])
    avg_v_peak = avg_v[idx]
    avg_i_peak = avg_i[idx]
    r = (avg_v_peak - avg_v_base) / (avg_i_peak - avg_i_base)

    return r


def get_square_pulse_idx(v):
    """
    Get up and down indices of the square pulse(s).
    Skipping the very first pulse (test pulse)

    Parameters
    ----------
    v: float
        pulse trace

    Returns
    -------
    up_idx, down_idx: list, list
        up, down indices
    """
    dv = np.diff(v)

    up_idx = np.flatnonzero(dv > 0)[1:] # skip the very first pulse (test pulse)
    down_idx = np.flatnonzero(dv < 0)[1:]

    assert len(up_idx) == len(down_idx), "Truncated square pulse"

    for up_ix, down_ix in zip(up_idx, down_idx):
        assert up_ix < down_ix, "Negative square pulse"

    return up_idx, down_idx


def average_pulses(v, i, t, cell_attached, access_cutoff_for_breakin=100, rmse_cutoff=50e-12):
    up_idx, down_idx = get_square_pulse_idx(v)
    if len(up_idx) == 0:
        return None, None, None, None, None

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    pulses_i_all = []
    pulses_i_no_spikes = []
    pulses_i_no_late_spikes = []
    pulses_v = []

    for u, d in zip(up_idx, down_idx):
        start_idx = u - one_ms * 2
        end_idx = d + one_ms * 2
        slice_t = dt * np.arange(end_idx - start_idx)
        relative_up_ind = u - start_idx
        relative_down_ind = d - start_idx

        end = relative_up_ind - 1
        start = end - one_ms
        i_baseline = np.mean(i[start_idx:end_idx][start:end])
        v_baseline = np.mean(v[start_idx:end_idx][start:end])

        # Check for noisy baseline
        rmse = np.sqrt(((i_baseline - i[start_idx:end_idx][start:end]) ** 2).mean())
        if rmse > rmse_cutoff:
            continue

        if cell_attached:
            # Check if it's already broken in
            peak_cap_ind = np.argmax(i[start_idx:end_idx][relative_up_ind:relative_up_ind + int(one_ms * 0.2)]) + relative_up_ind
            i_peak = i[start_idx:end_idx][peak_cap_ind]
            v_peak = v[start_idx:end_idx][peak_cap_ind]

            est_access_resistance = 1e6 * (v_peak - v_baseline) / (i_peak - i_baseline)
            if est_access_resistance < access_cutoff_for_breakin:
                break

        # check for contaminating spikes
        if cell_attached:
            spikes = detect_cell_attached_spikes(
                i[start_idx:end_idx][relative_up_ind:relative_down_ind] - i_baseline,
                slice_t[relative_up_ind:relative_down_ind])
        else:
            spikes = detect_escaped_spikes(
                i[start_idx:end_idx][relative_up_ind:relative_down_ind] - i_baseline,
                slice_t[relative_up_ind:relative_down_ind])

        pulses_v.append(v[start_idx:end_idx])
        pulses_i_all.append(i[start_idx:end_idx])
        if len(spikes) == 0:
            pulses_i_no_spikes.append(i[start_idx:end_idx])
        else:
            spike_inds = np.array([s[0] for s in spikes])
            if not np.any(spike_inds > relative_down_ind - relative_up_ind - 2 * one_ms):
                pulses_i_no_late_spikes.append(i[start_idx:end_idx])


    if len(pulses_i_no_spikes) > 0:
#         print(f"Using {len(pulses_i_no_spikes)} pulses with no spikes")
        avg_i = np.vstack(pulses_i_no_spikes).mean(axis=0)
    elif len(pulses_i_no_late_spikes) > 0:
#         print(f"Using {len(pulses_i_no_late_spikes)} pulses with no late spikes")
        avg_i = np.vstack(pulses_i_no_late_spikes).mean(axis=0)
    elif len(pulses_i_all) > 0:
#         print(f"Using {len(pulses_i_no_late_spikes)} pulses; all had late spikes")
        avg_i = np.vstack(pulses_i_all).mean(axis=0)
    else:
        avg_i = None

    if len(pulses_v) > 0:
        avg_v = np.vstack(pulses_v).mean(axis=0)
    else:
        avg_v = None

    return avg_i, avg_v, slice_t, relative_up_ind, relative_down_ind


def average_cell_attached_pulses(v, i, t, access_cutoff_for_breakin=100, rmse_cutoff=50e-12):
    return average_pulses(v, i, t,
        cell_attached=True,
        access_cutoff_for_breakin=access_cutoff_for_breakin,
        rmse_cutoff=rmse_cutoff
    )


def average_whole_cell_pulses(v, i, t, rmse_cutoff=50e-12):
    return average_pulses(v, i, t,
        cell_attached=False,
        rmse_cutoff=rmse_cutoff
    )


def detect_cell_attached_spikes(i, t, min_spike_amp = -30e-12):
    # i should be baselined
    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    # start detection after peak of capacitance transient
    cap_peak_ind = np.argmax(i[0:int(one_ms * 0.2)])

    putative_spikes = np.flatnonzero(np.diff(np.less_equal(i[cap_peak_ind:], min_spike_amp).astype(int)) == 1) + cap_peak_ind

    last_spike_t = t[0] - 0.001
    spikes = []
    for spike_ind in putative_spikes:
        if t[spike_ind] < last_spike_t + 0.001:
            # too close to previous spike
            continue

        # find peak in 1 ms window
        peak_ind = np.argmin(i[spike_ind:spike_ind + one_ms]) + spike_ind
        peak_amp = i[peak_ind]

        # check for biphasic - does it go at least 50% of min amplitude above baseline within a millisecond of negative-going peak?
        pos_peak_ind = np.argmax(i[peak_ind:peak_ind + one_ms]) + peak_ind
        pos_peak_amp = i[pos_peak_ind]
        if i[pos_peak_ind] >= 0.5 * -min_spike_amp:
            # count it as a spike
            spikes.append((peak_ind, peak_amp))
            last_spike_t = t[peak_ind]

    return spikes



def detect_escaped_spikes(i, t, min_spike_amp = -500e-12):
    # i should be baselined
    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    # start detection after peak of capacitance transient
    cap_peak_ind = np.argmax(i[0:int(one_ms * 0.2)])

    putative_spikes = np.flatnonzero(np.diff(np.less_equal(i[cap_peak_ind:], min_spike_amp).astype(int)) == 1) + cap_peak_ind

    last_spike_t = t[0] - 0.001
    spikes = []
    for spike_ind in putative_spikes:
        if t[spike_ind] < last_spike_t + 0.001:
            # too close to previous spike
            continue

        # find peak in 1 ms window
        peak_ind = np.argmin(i[spike_ind:spike_ind + one_ms]) + spike_ind
        peak_amp = i[peak_ind]
        spikes.append((peak_ind, peak_amp))

    return spikes


def _exp_curve(x, a, inv_tau, y0):
    return y0 + a * np.exp(-inv_tau * x)
