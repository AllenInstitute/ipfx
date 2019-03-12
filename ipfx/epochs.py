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
    Get epoch lasting LAST_STABILITY_EPOCH before idx1

    Parameters
    ----------
    idx1    : int last index of the epoch
    hz      : float sampling rate

    Returns
    -------
    (idx0,idx1) : int tuple of epoch indices

    """
    idx0 = idx1-int(POSTSTIM_STABILITY_EPOCH * hz)

    return idx0, idx1


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


def get_sweep_epoch(response):
    """
    Find sweep epoch defined as interval from the beginning of recordign to last non-zero value

    Parameters
    ----------
    response    : float np.array

    Returns
    -------
    (start_index,end_index): int tuple with start,end indices of the epoch

    """

    sweep_end_idx = np.flatnonzero(response)[-1]  # last non-zero index along the only dimension=0.

    return 0, sweep_end_idx


def get_stim_epoch_A(i,test_pulse=True):

    """
    Identify start index, and end index of a general stimulus.
    """

    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0

    start_idx_idx = 2 if test_pulse else 0     # skip the first up/down (test pulse) if present

    if len(di_idx[start_idx_idx:]) == 0:    # if no stimulus is found
        return None, None

    start_idx = di_idx[start_idx_idx] + 1   # shift by one to compensate for diff()
    end_idx = di_idx[-1]

    return start_idx, end_idx


def get_stim_epoch_B(i):
    """
    Identify start index, and end index of a general stimulus. Assume there is test pulse.
    """

    di = np.diff(i)
    di_idx = np.flatnonzero(di)  # != 0)

    if len(di_idx) == 0:
        return None, None

    if len(di_idx) >= 4:
        idx = 2  # skip the first up/down assuming that there is a test pulse
    else:
        idx = 0

    start_idx = di_idx[idx] + 1  # shift by one to compensate for diff()
    end_idx = di_idx[-1]

    return start_idx, end_idx


def get_stim_epoch(i,test_pulse=True):

    return get_stim_epoch_B(i)


def get_experiment_epoch(i,hz):
    """
    Find index range for the experiment epoch.
    The start index of the experiment epoch is defined as stim_start_idx - PRESTIM_DURATION*sampling_rate
    The end index of the experiment epoch is defined as stim_end_idx + POSTSTIM_DURATION*sampling_rate

    Parameters
    ----------
    i   :   float np.array of current
    hz  :   float sampling rate

    Returns
    -------
    (expt_start_idx,expt_end_idx): int tuple with start, end indices of the epoch

    """

    stim_start_idx, stim_end_idx = get_stim_epoch(i)
    if stim_start_idx and stim_end_idx:
        expt_start_idx = stim_start_idx - int(PRESTIM_STABILITY_EPOCH * hz)
        expt_end_idx = stim_end_idx + int(POSTSTIM_STABILITY_EPOCH * hz)

        return expt_start_idx,expt_end_idx
    else:
        return None, None


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


