import numpy as np


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

def measure_seal(v, curr, t):

    return 1e-9 * get_r_from_stable_pulse_response(v*1e-3, curr*1e-12, t)


def measure_input_resistance(v, curr, t):

    return 1e-6 * get_r_from_stable_pulse_response(v*1e-3, curr*1e-12, t)


def measure_initial_access_resistance(v, curr, t):
    return 1e-6 * get_r_from_peak_pulse_response(v*1e-3, curr*1e-12, t)


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


def get_r_from_stable_pulse_response(v, i, t):
    """Compute input resistance from the stable pulse response

    Parameters
    ----------
    v : float membrane voltage (V)
    i : float input current (A)
    t : time (s)

    Returns
    -------
    ir: float input resistance
    """

    up_idx, down_idx = get_square_pulse_idx(v)
    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    r = []

    # Average each pulse together, then calculate values
    # (This is less noisy than calculating the resistance separately
    # for each pulse)
    pulses_i = []
    pulses_v = []
    for u, d in zip(up_idx, down_idx):
        #
        start_idx = u - one_ms * 2
        end_idx = d + one_ms * 2
        pulses_i.append(i[start_idx:end_idx])
        pulses_v.append(v[start_idx:end_idx])

        relative_up_ind = u - start_idx
        relative_down_ind = d - start_idx

    avg_i = np.vstack(pulses_i).mean(axis=0)
    avg_v = np.vstack(pulses_v).mean(axis=0)

    # baseline - take average v and i one ms before start
    end = relative_up_ind - 1
    start = end - one_ms
    avg_v_base = np.mean(avg_v[start:end])
    avg_i_base = np.mean(avg_i[start:end])

    # steady-state - take average v and i one ms before end
    end = relative_down_ind - 1
    start = end - one_ms
    avg_v_steady = np.mean(avg_v[start:end])
    avg_i_steady = np.mean(avg_i[start:end])

    r = (avg_v_steady - avg_v_base) / (avg_i_steady - avg_i_base)

    return r


def get_r_from_peak_pulse_response(v, i, t):

    up_idx, down_idx = get_square_pulse_idx(v)

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    # Average each pulse together, then calculate values
    # (This is less noisy than calculating the resistance separately
    # for each pulse)
    pulses_i = []
    pulses_v = []
    for u, d in zip(up_idx, down_idx):
        #
        start_idx = u - one_ms * 2
        end_idx = d + one_ms * 2
        pulses_i.append(i[start_idx:end_idx])
        pulses_v.append(v[start_idx:end_idx])

        relative_up_ind = u - start_idx
        relative_down_ind = d - start_idx

    avg_i = np.vstack(pulses_i).mean(axis=0)
    avg_v = np.vstack(pulses_v).mean(axis=0)

    # take average v and i one ms before start of pulse
    end = relative_up_ind - 1
    start = end - one_ms
    avg_v_base = np.mean(v[start:end])
    avg_i_base = np.mean(i[start:end])

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
