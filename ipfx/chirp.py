from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack

from . import feature_vectors as fv
from . import time_series_utils as tsu


def extract_chirp_feature_vector(data_set, chirp_sweep_numbers):
    chirp_sweeps = data_set.sweep_set(chirp_sweep_numbers)

    features = feature_vectors_chirp(chirp_sweeps)
    return features


def feature_vectors_chirp(chirp_sweeps):
    result = {}
    amp, phase, freq = chirp_amp_phase(chirp_sweeps)
    result["chirp"] = np.hstack([amp, phase])
    return result


def chirp_amp_phase(sweep_set, start=0.6, end=20.6, down_rate=2000,
        min_freq=0.2, max_freq=40.):
    """ Calculate amplitude and phase of chirp responses

    Parameters
    ----------
    sweep_set: SweepSet
        Set of chirp sweeps
    start: float (optional, default 0.6)
        Start of chirp stimulus in seconds
    end: float (optional, default 20.6)
        End of chirp stimulus in seconds
    down_rate: int (optional, default 2000)
        Sampling rate for downsampling before FFT
    min_freq: float (optional, default 0.2)
        Minimum frequency for output to contain
    max_freq: float (optional, default 40)
        Maximum frequency for output to contain

    Returns
    -------
    amplitude: array
        Aka resistance
    phase: array
        Aka reactance
    freq: array
        Frequencies for amplitude and phase results
    """
    v_list = []
    i_list = []
    for swp in sweep_set.sweeps:
        # check for truncated sweep
        if np.all(swp.v[-100:] == 0):
            continue
        v_list.append(swp.v)
        i_list.append(swp.i)


    avg_v = np.vstack(v_list).mean(axis=0)
    avg_i = np.vstack(i_list).mean(axis=0)
    t = sweep_set.sweeps[0].t

    current_rate = np.rint(1 / (t[1] - t[0]))
    if current_rate > down_rate:
        width = int(current_rate / down_rate)
        ds_v = ds_v = fv.subsample_average(avg_v, width)
        ds_i = fv.subsample_average(avg_i, width)
        ds_t = t[::width]
    else:
        ds_v = avg_v
        ds_i = avg_i
        ds_t = t

    start_index = tsu.find_time_index(ds_t, start)
    end_index = tsu.find_time_index(ds_t, end)

    N = len(ds_v[start_index:end_index])
    T = ds_t[1] - ds_t[0]
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    v_fft = fftpack.fft(ds_v[start_index:end_index])
    i_fft = fftpack.fft(ds_i[start_index:end_index])
    Z = v_fft / i_fft
    R = np.real(Z)
    X = np.imag(Z)

    resistance = np.abs(Z)[0:N//2]
    reactance = np.arctan(X / R)[0:N//2]

    low_ind = tsu.find_time_index(xf, min_freq)
    high_ind = tsu.find_time_index(xf, max_freq)

    return resistance[low_ind:high_ind], reactance[low_ind:high_ind], xf[low_ind:high_ind]




