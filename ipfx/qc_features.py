import numpy as np


def measure_blowout(v, idx0):

    return 1e3 * np.mean(v[idx0:]*1e-3)


def measure_electrode_0(curr, hz, t=0.005):

    n_time_steps = int(t * hz)
    # electrode 0 is the average current reading with zero voltage input
    # (ie, the equivalent of resting potential in current-clamp mode)
    return 1e12 * np.mean(curr[0:n_time_steps]*1e-12)


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
    return mean, rms
########################################################################


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


    dv = np.diff(v)
    up_idx = np.flatnonzero(dv > 0)
    down_idx = np.flatnonzero(dv < 0)
    assert len(up_idx) == len(down_idx), "Truncated breakin sweep, truncated response to a square pulse"
    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    r = []
    for ii in range(len(up_idx)):
        # take average v and i one ms before start
        end = up_idx[ii] - 1
        start = end - one_ms

        avg_v_base = np.mean(v[start:end])
        avg_i_base = np.mean(i[start:end])

        # take average v and i one ms before end
        end = down_idx[ii]-1
        start = end - one_ms

        avg_v_steady = np.mean(v[start:end])
        avg_i_steady = np.mean(i[start:end])

        r_instance = (avg_v_steady-avg_v_base) / (avg_i_steady-avg_i_base)

        r.append(r_instance)

    return np.mean(r)


def get_r_from_peak_pulse_response(v, i, t):
    dv = np.diff(v)
    up_idx = np.flatnonzero(dv > 0)
    down_idx = np.flatnonzero(dv < 0)
    assert len(up_idx) == len(down_idx), "Truncated breakin sweep, truncated response to a square pulse"

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)
    r = []
    for ii in range(len(up_idx)):
        # take average v and i one ms before
        end = up_idx[ii] - 1
        start = end - one_ms
        avg_v_base = np.mean(v[start:end])
        avg_i_base = np.mean(i[start:end])
        # take average v and i one ms before end
        start = up_idx[ii]
        end = down_idx[ii] - 1
        idx = start + np.argmax(i[start:end])
        avg_v_peak = v[idx]
        avg_i_peak = i[idx]
        r_instance = (avg_v_peak-avg_v_base) / (avg_i_peak-avg_i_base)
        r.append(r_instance)

    return np.mean(r)
