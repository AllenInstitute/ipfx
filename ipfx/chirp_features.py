from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack
from scipy.signal import savgol_filter

from ipfx.error import FeatureError
from . import feature_vectors as fv
from . import time_series_utils as tsu

CHIRP_CODES = [
            "C2CHIRP180503", # current version, single length
            "C2CHIRP171129", # early version, three lengths
            "C2CHIRP171103", # only one example
        ]


def extract_chirp_features(sweepset, **params):
    results = []
    for sweep in sweepset.sweeps:
        try:
            result = chirp_sweep_features(sweep, **params)
            results['sweep_number'] = sweep.sweep_number
            results.append(result)
        except FeatureError as exc:
            logging.debug(exc)
        except Exception:
            msg = "Error processing chirp sweep {} for specimen {:d}.".format(sweep.sweep_number, specimen_id)
            logging.warning(msg, exc_info=True)

    if len(results)==0:
        logging.debug("No chirp sweep results for specimen {:d}.".format(specimen_id))
        return {}

    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    mean_results['sweeps'] = results
    return mean_results

def chirp_sweep_amp_phase(sweep, min_freq=1., max_freq=35., smooth_bw=2, raw=False, **transform_params):
    """ Calculate amplitude and phase of chirp response

    Parameters
    ----------
    sweep_set: Sweep
        Set of chirp sweeps
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
    v, i, freq = transform_sweep(sweep, **transform_params)
    Z = v / i
    amp = np.abs(Z)
    phase = np.angle(Z)
    if raw:
        return amp, phase, freq
    
    # window before or after smoothing?
    low_ind = tsu.find_time_index(xf, min_freq)
    high_ind = tsu.find_time_index(xf, max_freq)
    amp, phase, freq = map(lambda x: x[low_ind:high_ind], [amp, phase, freq])
    
    # pick odd number, approx number of points for smooth_bw interval
    n_filt = int(np.rint(1/(freq[1]-freq[0])))*smooth_bw + 1
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
    v = fv._subsample_average(np.pad(v, (pad,0), 'constant', constant_values=np.nan), width)
    i = fv._subsample_average(np.pad(i, (pad,0), 'constant', constant_values=np.nan), width)
    dt = t[width] - t[0]
    xf = np.linspace(0.0, 1.0/(2.0*dt), len(v)//2)

    v_fft = fftpack.fft(v)
    i_fft = fftpack.fft(i)

    return v_fft, i_fft, xf

def chirp_sweep_features(sweep, method_params={}):
    amp, phase, freq = chirp_sweep_amp_phase(sweep, method_params=method_params)
    i_max = np.argmax(amp)
    z_max = amp[i_max]
    i_cutoff = np.argmin(abs(amp - z_max/np.sqrt(2)))
    features = {
        "peak_ratio": amp[i_max]/amp[0],
        "peak_freq": freq[i_max],
        "3db_freq": freq[i_cutoff],
        "z_low": amp[0],
        "z_high": amp[-1],
        "z_peak": z_max,
        "phase_peak": phase[i_max],
        "phase_low": phase[0],
        "phase_high": phase[-1]
    }
    return features

