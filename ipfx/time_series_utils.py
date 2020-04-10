import sys

import numpy as np
import scipy.signal as signal


def find_time_index(t, t_0):
    """ Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """
    assert t[0] <= t_0 <= t[-1], "Given time ({:f}) is outside of time range ({:f}, {:f})".format(t_0, t[0], t[-1])

    idx = np.argmin(abs(t - t_0))
    return idx


def calculate_dvdt(v, t, filter=None):
    """Low-pass filters (if requested) and differentiates voltage by time.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default None)

    Returns
    -------
    dvdt : numpy array of time-derivative of voltage (V/s = mV/ms)
    """

    if has_fixed_dt(t) and filter:
        delta_t = t[1] - t[0]
        sample_freq = 1. / delta_t
        filt_coeff = (filter * 1e3) / (sample_freq / 2.) # filter kHz -> Hz, then get fraction of Nyquist frequency
        if filt_coeff < 0 or filt_coeff >= 1:
            raise ValueError("bessel coeff ({:f}) is outside of valid range [0,1); cannot filter sampling frequency {:.1f} kHz with cutoff frequency {:.1f} kHz.".format(filt_coeff, sample_freq / 1e3, filter))
        b, a = signal.bessel(4, filt_coeff, "low")
        v_filt = signal.filtfilt(b, a, v, axis=0)
        dv = np.diff(v_filt)
    else:
        dv = np.diff(v)

    dt = np.diff(t)
    dvdt = 1e-3 * dv / dt  # in V/s = mV/ms

    # some data sources, such as neuron, occasionally report 
    # duplicate timestamps, so we require that dt is not 0
    return dvdt[np.fabs(dt) > sys.float_info.epsilon]


def has_fixed_dt(t):
    """Check that all time intervals are identical."""
    dt = np.diff(t)
    return np.allclose(dt, np.ones_like(dt) * dt[0])


def average_voltage(v, t, start=None, end=None):
    """Calculate average voltage between start and end.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional, default None)
    end : end of time window for spike detection (optional, default None)

    Returns
    -------
    v_avg : average voltage
    """

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = find_time_index(t, start)
    end_index = find_time_index(t, end)

    return v[start_index:end_index].mean()


def flatnotnan(a):
    """
    Returns indices that are non nan in a flattened version of a

    Parameters
    ----------
    a: np.array

    Returns
    -------
    res: np.array
        Output array containing indices of an array that are not nan
    """
    notnan = np.argwhere(~np.isnan(a))

    return notnan.flatten()


