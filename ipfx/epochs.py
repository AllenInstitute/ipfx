import numpy as np
import ipfx.time_series_utils as tsu


# global constants
#TODO: read them from the config file

NOISE_EPOCH = 0.0015
PRESTIM_STABILITY_EPOCH = 0.5
POSTSTIM_STABILITY_EPOCH = 0.5
TEST_PULSE_MAX_TIME = 0.4


def get_first_stability_epoch(stim_start_idx, hz):

    num_steps = int(PRESTIM_STABILITY_EPOCH * hz)
    if num_steps > stim_start_idx-1:
        num_steps = stim_start_idx-1
    elif num_steps <= 0:
        return 0, 0

    return stim_start_idx-1-num_steps, stim_start_idx-1


def get_last_stability_epoch(idx1, hz):
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


def get_first_noise_epoch(idx, hz):

    return idx, idx + int(NOISE_EPOCH * hz)


def get_last_noise_epoch(idx1, hz):

    return idx1-int(NOISE_EPOCH * hz), idx1


def get_recording_epoch(response):
    """
    Detect response epoch defined as interval from start to the last non-nan value of the response

    Parameters
    ----------
    response: float np.array

    Returns
    -------
    start,end: int
        indices of the epoch
    """

    if len(tsu.flatnotnan(response)) == 0:
        end_idx = 0
    else:
        end_idx = tsu.flatnotnan(response)[-1]
    return 0, end_idx


def get_sweep_epoch(response):
    """
    Defined as interval including entire sweep

    Parameters
    ----------
    response: float np.array

    Returns
    -------
    (start_index,end_index): int tuple
        with start,end indices of the epoch

    """

    return 0, len(response)-1


def get_stim_epoch(i, test_pulse=True):
    """
    Determine the start index, and end index of a general stimulus.

    Parameters
    ----------
    i   : numpy array
        current

    test_pulse: bool
        True if test pulse is assumed

    Returns
    -------
    start,end: int tuple
    """

    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0

    if test_pulse:
        di_idx = di_idx[2:]     # drop the first up/down (test pulse) if present

    if len(di_idx) == 0:    # if no stimulus is found
        return None

    start_idx = di_idx[0] + 1   # shift by one to compensate for diff()
    end_idx = di_idx[-1]

    return start_idx, end_idx


def get_test_epoch(i,hz):
    """
    Find index range of the test epoch

    Parameters
    ----------
    i : float np.array
        current trace

    Returns
    -------
    start_idx,end_idx: int tuple
        start,end indices of the epoch
    hz: float
        sampling rate
    """

    di = np.diff(i)
    di_idx = np.flatnonzero(di)

    if len(di_idx) == 0:
        return None

    if di_idx[0] >= TEST_PULSE_MAX_TIME*hz:
        return None

    if len(di_idx) == 1:
        raise Exception("Cannot detect and end to the test pulse")

    start_pulse_idx = di_idx[0] + 1  # shift by one to compensate for diff()
    end_pulse_idx = di_idx[1]
    padding = start_pulse_idx

    start_idx = start_pulse_idx - padding
    end_idx = end_pulse_idx + padding

    return start_idx, end_idx


def get_experiment_epoch(i, hz, test_pulse=True):
    """
    Find index range for the experiment epoch.
    The start index of the experiment epoch is defined as stim_start_idx - PRESTIM_DURATION*sampling_rate
    The end index of the experiment epoch is defined as stim_end_idx + POSTSTIM_DURATION*sampling_rate

    Parameters
    ----------
    i   :   float np.array of current
    hz  :   float sampling rate
    test_pulse: bool True if present, False otherwise

    Returns
    -------
    (expt_start_idx,expt_end_idx): int tuple with start, end indices of the epoch

    """

    stim_epoch = get_stim_epoch(i,test_pulse)

    if stim_epoch:
        stim_start_idx,stim_end_idx = stim_epoch
        expt_start_idx = stim_start_idx - int(PRESTIM_STABILITY_EPOCH * hz)
        expt_end_idx = stim_end_idx + int(POSTSTIM_STABILITY_EPOCH * hz)

        return expt_start_idx,expt_end_idx
    else:
        return None


