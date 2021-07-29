import numpy as np
import scipy
import ipfx.chirp as chirp
from ipfx.sweep import Sweep, SweepSet


def test_chirp_output():
    # Stuff for sweep construction
    sampling_rate = 2000
    t = np.arange(0, 20 * sampling_rate) * (1 / sampling_rate)
    clamp_mode = "CurrentClamp"
    epochs = {"recording": (0, len(t) - 1),
        "test": None,
        "experiment": None,
        "stim": None}

    base_chirp = scipy.signal.chirp(t, 0.5, 20, 40, method="linear")
    i = base_chirp

    # linear descreasing profile
    profile = np.linspace(1., 0.5, num=len(t))
    v = base_chirp * profile
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    sweep_set = SweepSet([test_sweep])

    amp, phase, freq = chirp.chirp_amp_phase(sweep_set, start=0, end=19.9)

    # Confirm goes from 1 to 0.5
    tol = 0.1
    assert np.abs(amp[0] - 1) < tol
    assert np.abs(amp[-1] - 0.5) < tol

    # "resonant" profile
    profile = (-(t - 10) ** 2 + 100) / 100 + 1
    v = base_chirp * profile
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    sweep_set = SweepSet([test_sweep])

    amp, phase, freq = chirp.chirp_amp_phase(sweep_set, start=0, end=19.9)

    # Confirm it peaks around 2 near 20 Hz
    amp_tol = 0.1
    freq_tol = 1

    assert np.abs(np.max(amp) - 2) < amp_tol
    assert np.abs(freq[np.argmax(amp)] - 20) < freq_tol


def test_chirp_downsample():
    # Stuff for sweep construction
    sampling_rate = 2000
    t = np.arange(0, 20 * sampling_rate) * (1 / sampling_rate)
    clamp_mode = "CurrentClamp"
    epochs = {"recording": (0, len(t) - 1),
        "test": None,
        "experiment": None,
        "stim": None}

    base_chirp = scipy.signal.chirp(t, 0.5, 20, 40, method="linear")
    i = base_chirp

    # linear descreasing profile
    profile = np.linspace(1., 0.5, num=len(t))
    v = base_chirp * profile
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    sweep_set = SweepSet([test_sweep])

    amp, phase, freq = chirp.chirp_amp_phase(sweep_set, start=0, end=19.9, down_rate=sampling_rate/2)

    # Confirm goes from 1 to 0.5
    tol = 0.1
    assert np.abs(amp[0] - 1) < tol
    assert np.abs(amp[-1] - 0.5) < tol


def test_divide_chirps_by_stimulus():
    # Stuff for sweep construction
    sampling_rate = 2000
    t = np.arange(0, 20 * sampling_rate) * (1 / sampling_rate)
    clamp_mode = "CurrentClamp"
    epochs = {"recording": (0, len(t) - 1),
        "test": None,
        "experiment": None,
        "stim": None}

    i1 = scipy.signal.chirp(t, 0.5, 20, 40, method="linear")
    i2 = scipy.signal.chirp(t, 0.5, 20, 40, method="logarithmic")
    # linear descreasing profile
    profile = np.linspace(1., 0.5, num=len(t))

    sweep_set = SweepSet([Sweep(t, i * profile, i, clamp_mode, sampling_rate, epochs=epochs)
        for i in (i1, i2, i2)])

    divided_list = chirp.divide_chirps_by_stimulus(sweep_set)
    lengths = [len(d.sweeps) for d in divided_list]
    assert np.min(lengths) == 1
    assert np.max(lengths) == 2
    bigger_sweep_set = divided_list[np.argmax(lengths)]
    smaller_sweep_set = divided_list[np.argmin(lengths)]
    assert np.all(bigger_sweep_set.sweeps[0].i == bigger_sweep_set.sweeps[1].i)
    assert np.any(bigger_sweep_set.sweeps[0].i != smaller_sweep_set.sweeps[0].i)



