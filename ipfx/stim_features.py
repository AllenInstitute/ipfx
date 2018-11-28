import numpy as np
import time_series_utils as tsu


# global constants
#TODO: read them from the config file

NOISE_EPOCH = 0.0015
PRESTIM_STABILITY_EPOCH = 0.5
POSTSTIM_STABILITY_EPOCH = 0.5
LONG_RESPONSE_DURATION = 5  # this will count long ramps as completed


def get_last_vm_epoch(idx1, hz):
    """
    Get epoch lasting LAST_STABILITY_EPOCH before the end of recording

    Parameters
    ----------
    idx1
    hz

    Returns
    -------

    """
    return idx1-int(POSTSTIM_STABILITY_EPOCH * hz), idx1


def get_first_vm_noise_epoch(idx, hz):

    return idx, idx + int(NOISE_EPOCH * hz)


def get_last_vm_noise_epoch(idx1, hz):

    return idx1-int(NOISE_EPOCH * hz), idx1


def get_stability_vm_epoch(stim_start, hz):
    num_steps = int(PRESTIM_STABILITY_EPOCH * hz)
    if num_steps > stim_start-1:
        num_steps = stim_start-1
    elif num_steps <= 0:
        return 0, 0
    assert num_steps > 0, "Number of steps should be a positive integer"

    return stim_start-1-num_steps, stim_start-1


def find_stim_start(stim, idx0=0):
    """
    Find the index of the first nonzero positive or negative jump in an array.

    Parameters
    ----------
    stim: np.ndarray
        Array to be searched

    idx0: int
        Start searching with this index (default: 0).

    Returns
    -------
    int
    """

    di = np.diff(stim)
    idxs = np.flatnonzero(di)
    idxs = idxs[idxs >= idx0]

    if len(idxs) == 0:
        return -1

    return idxs[0]+1


def get_recording_end_idx(v):

    end_idx = np.nonzero(v)[0][-1]  # last non-zero index along the only dimension=0.
    return end_idx

def get_sweep_epoch(response):

    sweep_end_idx = np.nonzero(response)[0][-1]  # last non-zero index along the only dimension=0.

    return (0, sweep_end_idx)

def get_experiment_epoch(i,v,hz):
    """
    Find index range for the experiment epoch. The start is defined as stim start- PRESTIM_DURATION*sampling_rate
    The end is defined by the last nonzero response.


    Parameters
    ----------
    i : stimulus
    v : response
    hz: sampling rate

    Returns
    -------
    start and end indices

    """
    # TODO: deal with non iclamp sweeps and non experimental sweeps
    di = np.diff(i)
    diff_idx = np.flatnonzero(di)  # != 0)

    if len(diff_idx) == 0:
        raise ValueError("Empty stimulus trace")
    if len(diff_idx) >= 4:
        idx = 2  # skip the first up/down assuming that there is a test pulse
    else:
        idx = 0

    stim_start_idx = diff_idx[idx] + 1  # shift by one to compensate for diff()
    expt_start_idx = stim_start_idx - int(PRESTIM_STABILITY_EPOCH * hz)
    #       Recording ends when zeros start
    expt_end_idx = np.nonzero(v)[0][-1]  # last non-zero index along the only dimension=0.

    return expt_start_idx,expt_end_idx


def find_stim_window(stim, idx0=0):
    """
    Find the index of the first nonzero positive or negative jump in an array and the number of timesteps until the last such jump.

    Parameters
    ----------
    stim: np.ndarray
        Array to be searched

    idx0: int
        Start searching with this index (default: 0).

    Returns
    -------
    start_index, duration
    """

    di = np.diff(stim)
    idxs = np.flatnonzero(di)
    idxs = idxs[idxs >= idx0]

    if len(idxs) == 0:
        return -1, 0

    stim_start = idxs[0]+1

    if len(idxs) == 1:
        return stim_start, len(stim)-stim_start

    stim_end = idxs[-1]+1

    return stim_start, stim_end - stim_start


def find_stim_amplitude_and_duration(idx0, stim, hz):
    stim=np.array(stim)
    if len(stim) < idx0:
        idx0 = 0

    stim = stim[idx0:]

    peak_high = max(stim)
    peak_low = min(stim)

    # measure stimulus length
    # find index of first non-zero value, and last return to zero
    nzero = np.where(stim!=0)[0]
    if len(nzero) > 0:
        start = nzero[0]
        end = nzero[-1]+1
        dur = (end - start) / hz
    else:
        dur = 0

    dur = float(dur)

    if abs(peak_high) > abs(peak_low):
        amp = float(peak_high)
    else:
        amp = float(peak_low)
    return amp, dur


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


def get_stim_characteristics(i, t, no_test_pulse=False):
    """
    Identify the start time, duration, amplitude, start index, and
    end index of a general stimulus.
    This assumes that there is a test pulse followed by the stimulus square.
    """

    di = np.diff(i)
    diff_idx = np.flatnonzero(di)# != 0)

    if len(diff_idx) == 0:
        return (None, None, 0.0, None, None)

    # skip the first up/down
    idx = 0 if no_test_pulse else 2

    # shift by one to compensate for diff()
    start_idx = diff_idx[idx] + 1
    end_idx = diff_idx[-1] + 1

    stim_start = float(t[start_idx])
    stim_dur = float(t[end_idx] - t[start_idx])
    stim_amp = float(i[start_idx])

    return (stim_start, stim_dur, stim_amp, start_idx, end_idx)


def _step_stim_amp(t, i, start):
    t_index = tsu.find_time_index(t, start)
    return i[t_index + 1]


def _short_step_stim_amp(t, i, start):
    t_index = tsu.find_time_index(t, start)
    return i[t_index + 1:].max()
