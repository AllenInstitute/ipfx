import numpy as np
from . import time_series_utils as tsu


def get_stim_characteristics(i, t, test_pulse=True):
    """
    Identify the start time, duration, amplitude, start index, and end index of a general stimulus.
    """

    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0
    start_idx_idx = 2 if test_pulse else 0     # skip the first up/down (test pulse) if present

    if len(di_idx[start_idx_idx:]) == 0:    # if no stimulus is found
        return None, None, 0.0, None, None

    start_idx = di_idx[start_idx_idx] + 1   # shift by one to compensate for diff()
    end_idx = di_idx[-1]
    if start_idx >= end_idx: # sweep has been cut off before stimulus end
        return None, None, 0.0, None, None

    start_time = float(t[start_idx])
    duration = float(t[end_idx] - t[start_idx-1])

    stim = i[start_idx:end_idx+1]

    peak_high = max(stim)
    peak_low = min(stim)

    if abs(peak_high) > abs(peak_low):
        amplitude = float(peak_high)
    else:
        amplitude = float(peak_low)

    return start_time, duration, amplitude, start_idx, end_idx


def _step_stim_amp(t, i, start):
    t_index = tsu.find_time_index(t, start)
    return i[t_index + 1]


def _short_step_stim_amp(t, i, start):
    t_index = tsu.find_time_index(t, start)
    return i[t_index + 1:].max()

def find_stim_interval(idx0, stim, hz):
    stim = np.array(stim)[idx0:]

    # indices where is the stimulus off
    zero_idxs = np.where(stim == 0)[0]

    # derivative of off indices.  when greater than one, indicates on period
    dzero_idxs = np.diff(zero_idxs)
    dzero_break_idxs = np.where(dzero_idxs[:] > 1)[0]

    # duration of breaks
    break_durs = dzero_idxs[dzero_break_idxs]

    # indices of breaks
    break_idxs = zero_idxs[dzero_break_idxs] + 1

    # time between break onsets
    dbreaks = np.diff(break_idxs)
    if len(np.unique(break_durs)) == 1 and len(np.unique(dbreaks)) == 1:
        return dbreaks[0] / hz

    return None
