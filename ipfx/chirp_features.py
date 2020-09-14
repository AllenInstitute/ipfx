from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from ipfx.error import FeatureError
from ipfx.subthresh_features import baseline_voltage
from . import feature_vectors as fv
from . import time_series_utils as tsu

CHIRP_CODES = [
            "C2CHIRP180503", # current version, single length
            "C2CHIRP171129", # early version, three lengths
            "C2CHIRP171103", # only one example
        ]


def extract_chirp_features_by_sweep(sweepset, **params):
    results = []
    for sweep in sweepset.sweeps:
        try:
            amp, phase, freq = chirp_sweep_amp_phase(sweep, **params)
            result = chirp_sweep_features(amp, phase, freq)
            result['sweep_number'] = sweep.sweep_number
            results.append(result)
        except FeatureError as exc:
            logging.debug(exc)
        except Exception:
            msg = F"Error processing chirp sweep {sweep.sweep_number}."
            logging.warning(msg, exc_info=True)

    if len(results)==0:
        logging.warning("No chirp sweep results available.")
        return {}

    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    mean_results['sweeps'] = results
    return mean_results

def extract_chirp_features(sweepset, **params):
    amps = []
    min_freq=None
    max_freq=None
    for i, sweep in enumerate(sweepset.sweeps):
        try:
            amp, freq = amp_response_asymmetric(sweep, min_freq=min_freq, max_freq=max_freq, **params)
            amps.append(amp)
            # apply the frequency bounds from the first sweep to the others
            if min_freq is None:
                min_freq = freq[0]
                max_freq = freq[-1]
        except FeatureError as exc:
            logging.warning(exc)
    if len(amps)==0:
        raise FeatureError('No valid chirp sweeps available.')
    amp = np.stack(amps).mean(axis=0)
    results = chirp_sweep_features(amp, freq, low_freq_max=1)
    return results

def amp_response_asymmetric(sweep, min_freq=None, max_freq=None, n_freq=500, freq_sigma=0.25):
    width = 8
    sweep.align_to_start_of_epoch('stim')
    sweep.select_epoch('experiment')
    v0 = baseline_voltage(sweep.t, sweep.v, start=0)
    sweep.select_epoch('stim')
    t = sweep.t

    v = tsu.subsample_average(sweep.v, width)
    i = tsu.subsample_average(sweep.i, width)

    fs = 1 / (sweep.t[1] * width)

    i_crossings = np.nonzero(np.diff(i>0))[0]
    i_peaks = np.array([np.argmax(np.abs(i[j:k])) + j for j, k in
                        zip(i_crossings[:-1], i_crossings[1:])])
    i_freq = fs/(2*np.diff(i_crossings))
    freq_fcn = interp1d(i_peaks, i_freq, assume_sorted=True, kind=2, )
    
    v = (v-v0)
    v_crossings = np.nonzero(np.diff(v>0))[0]
#     filter out noise crossings
    v_crossings = np.delete(v_crossings, np.nonzero(np.diff(v_crossings)<5))
    v_peaks = np.array([np.argmax(np.abs(v[j:k])) + j for j, k in
                        zip(v_crossings[:-1], v_crossings[1:])])
    
    v_peaks = v_peaks[(i_peaks[0] <= v_peaks) & (i_peaks[-1] >= v_peaks)]
    v_freq = freq_fcn(v_peaks)
    amp = np.abs(v[v_peaks])/np.max(i)
    upper = (v[v_peaks]>0)
    lower = (v[v_peaks]<0)
    
    freq = np.linspace(min_freq or i_freq[0], max_freq or i_freq[-1], n_freq)
    amp_upper = gauss_smooth(v_freq[upper], amp[upper], freq, freq_sigma)
    amp_lower = gauss_smooth(v_freq[lower], amp[lower], freq, freq_sigma)
    amp = np.stack([amp_upper, amp_lower]).mean(axis=0)
    return amp, freq

def chirp_sweep_amp_phase(sweep, min_freq=0.4, max_freq=40.0, filter_bw=2, filter=True, **transform_params):
    """ Calculate amplitude and phase of chirp response

    Parameters
    ----------
    sweep_set: Sweep
        Set of chirp sweeps
    min_freq: float (optional, default 0.4)
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
    v, i, freq = transform_sweep(sweep, **transform_params)
    Z = v / i
    amp = np.abs(Z)
    phase = np.angle(Z)
    
    # window before or after smoothing?
    low_ind = tsu.find_time_index(freq, min_freq)
    high_ind = tsu.find_time_index(freq, max_freq)
    amp, phase, freq = map(lambda x: x[low_ind:high_ind], [amp, phase, freq])
    
    if filter:
        # pick odd number, approx number of points for smooth_bw interval
        n_filt = int(np.rint(filter_bw/2/(freq[1]-freq[0])))*2 + 1
        filt = lambda x: savgol_filter(x, n_filt, 5)
        amp, phase = map(filt, [amp, phase])

    return amp, phase, freq

def transform_sweep(sweep, n_sample=10000):
    """ Calculate Fourier transform of sweep current and voltage
    """
    sweep.select_epoch("stim")
    if np.all(sweep.v[-10:] == 0):
        raise FeatureError("Chirp stim epoch truncated.")
    v = sweep.v
    i = sweep.i
    t = sweep.t
    N = len(v)

    width = int(N / n_sample)
    pad = int(width*np.ceil(N/width) - N)
    v = tsu.subsample_average(v, width)
    i = tsu.subsample_average(i, width)
    dt = t[width] - t[0]

    nfreq = len(v)//2
    freq = np.linspace(0.0, 1.0/(2.0*dt), nfreq)

    v_fft = fftpack.fft(v)
    i_fft = fftpack.fft(i)

    return v_fft[:nfreq], i_fft[:nfreq], freq

def chirp_sweep_features(amp, freq, low_freq_max=1.5):
    """Calculate a set of characteristic features from the impedance amplitude profile (ZAP).
    Peak response is measured relative to a low-frequency average response.

    Args:
        amp (ndarray): impedance amplitude (generalized resistance)
        freq (ndarray): frequencies corresponding to amp responses
        low_freq_max (float, optional): Upper frequency cutoff for low-frequency average reference value. Defaults to 1.5.

    Returns:
        dict: features
    """    
    i_max = np.argmax(amp)
    z_max = amp[i_max]
    i_cutoff = np.argmin(abs(amp - z_max/np.sqrt(2)))
    low_freq_amp = np.mean(amp[freq < low_freq_max])
    features = {
        "peak_ratio": z_max/low_freq_amp,
        "peak_freq": freq[i_max],
        "3db_freq": freq[i_cutoff],
        "r_low": low_freq_amp,
        "r_peak": z_max,
        # "r_high": amp[-1],
        # "phase_peak": phase[i_max],
        # "phase_low": phase[0],
        # "phase_high": phase[-1]
    }
    return features

def gauss_smooth(x, y, x_eval, sigma):
    """Apply smoothing to irregularly sampled data using a gaussian kernel

    Args:
        x (ndarray): 1D array of points y is sampled at
        y (ndarray): 1D array of data to be smoothed
        x_eval (ndarray): 1D array of points to evaluate smoothed function at
        sigma (float): standard deviation of gaussian kernel

    Returns:
        ndarray: 1D array of smoothed data evaluated at x_eval
    """    
    delta_x = x_eval[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)
    return y_eval
