from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack

from . import offpipeline_utils as op
from . import feature_vectors as fv
from . import time_series_utils as tsu

CHIRP_CODES = [
            "C2CHIRP180503",
            "C2CHIRP171129",
            "C2CHIRP171103",
        ]

def extract_chirp_features(specimen_id, data_source='lims', sweep_qc_option='none', **kwargs):
    dataset = op.dataset_for_specimen_id(specimen_id, data_source=data_source, ontology=ontology_with_chirps())
    sweepset = op.sweepset_by_type_qc(dataset, specimen_id, stimuli_names=["Chirp"])
    results = []
    if len(sweepset.sweeps)==0:
        return {}
    for sweep in sweepset.sweeps:
        results.append(chirp_sweep_features(sweep, **kwargs))
    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    return mean_results

def chirp_sweep_features(sweep, **kwargs):
    v, i, freq = transform_sweep(sweep, **kwargs)
    Z = v / i
    amp = np.abs(Z)
    phase = np.angle(Z)

    from scipy.signal import savgol_filter
    # pick odd number, approx number of points for 2 Hz interval
    n_filt = int(np.rint(1/(freq[1]-freq[0])))*2 + 1
    filt = lambda x: savgol_filter(x, n_filt, 5)
    amp, phase = map(filt, [amp, phase])

    imax = np.argmax(amp)
    features = {
        "z_low": amp[0],
        "z_high": amp[-1],
        "z_max": amp[imax],
        "peak_ratio": amp[imax]/amp[0],
        "peak_freq": freq[imax],
        "phase_max": np.max(phase),
        "phase_low": phase[0],
        "phase_high": phase[-1]
    }
    return features

def extract_chirp_feature_vector(data_set, chirp_sweep_numbers):
    chirp_sweeps = data_set.sweep_set(chirp_sweep_numbers)
    result = {}
    amp, phase, freq = chirp_amp_phase(chirp_sweeps)
    result["chirp"] = np.hstack([amp, phase])
    return features


def averaged_chirp_transform(sweep_set, df=0.1, n_sample=10000):
    """ Calculate averaged amplitude and phase of chirp responses,
    averaging results after transform.
    """
    amp_list = []
    phase_list = []
    for sweep in sweep_set.sweeps:
        v, i, freq = transform_sweep(sweep, n_sample=n_sample)
        Z = v / i
        amp = np.abs(Z)
        phase = np.angle(Z)
        
        width = int(np.rint(df/(freq[1]-freq[0])))
        amp = fv._subsample_average(amp, width)
        phase = fv._subsample_average(phase, width)
        amp_list.append(amp)
        phase_list.append(phase)

    avg_amp = np.vstack(amp_list).mean(axis=0)
    avg_phase = np.vstack(phase_list).mean(axis=0)
    return avg_amp, avg_phase, freq

def chirp_amp_phase(t, v, i, start=0.6, end=20.6, down_rate=2000,
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
        ds_v = ds_v = fv._subsample_average(avg_v, width)
        ds_i = fv._subsample_average(avg_i, width)
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


def transform_sweep(sweep, n_sample=10000, min_freq=1., max_freq=35.):
    sweep.select_epoch("stim")
    v = sweep.v
    i = sweep.i
    t = sweep.t
    N = len(v)

    # down_rate=2000
    # width = int(sweep.sampling_rate / down_rate)

    width = int(N / n_sample)
    pad = int(width*np.ceil(N/width) - N)
    v = fv._subsample_average(np.pad(v, (pad,0), 'constant', constant_values=np.nan), width)
    i = fv._subsample_average(np.pad(i, (pad,0), 'constant', constant_values=np.nan), width)
    t = t[::width]

    N = len(v)
    dt = t[1] - t[0]
    xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)

    v_fft = fftpack.fft(v)
    i_fft = fftpack.fft(i)

    low_ind = tsu.find_time_index(xf, min_freq)
    high_ind = tsu.find_time_index(xf, max_freq)

    return v_fft[low_ind:high_ind], i_fft[low_ind:high_ind], xf[low_ind:high_ind]


import json
import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
def ontology_with_chirps(chirp_stimulus_codes=CHIRP_CODES):
# Manual edit ontology to identify chirp sweeps
    ontology_data = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)
    mask = []
    for od in ontology_data:
        mask_val = True
        for tagset in od:
            for c in chirp_stimulus_codes:
                if c in tagset and "code" in tagset:
                    mask_val = False
                    break
        mask.append(mask_val)
    ontology_data = [od for od, m in zip(ontology_data, mask) if m is True]
    ontology_data.append([
        ["code"] + chirp_stimulus_codes,
        [
          "name",
          "Chirp",
        ],
        [
          "core",
          "Core 2"
        ]
    ])
    return StimulusOntology(ontology_data)
