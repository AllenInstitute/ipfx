import ipfx.subthresh_features as subf
import numpy as np


def test_input_resistance():
    t = np.arange(0, 1.0, 5e-6)
    v1 = np.ones_like(t) * -5.
    v2 = np.ones_like(t) * -10.
    i1 = np.ones_like(t) * -50.
    i2 = np.ones_like(t) * -100.

    ri = subf.input_resistance([t, t], [i1, i2], [v1, v2], 0, t[-1])

    assert np.allclose(ri, 100.)


def test_time_constant():
    dt = 5e-6
    baseline = -70.
    t = np.arange(0, 3.0, dt)
    v = np.ones_like(t) * baseline
    i = np.ones_like(t) * 0.
    start = 1.
    end = 2.
    start_index = int(start / dt)
    end_index = int(end / dt)
    actual_tau = 0.02
    A = 10.

    v[start_index:end_index] = (baseline - A) + A * np.exp(-(t[start_index:end_index] - t[start_index]) / actual_tau)

    tau = subf.time_constant(t, v, i, start=start, end=end)
    assert np.isclose(actual_tau, tau)


def test_time_constant_noise_rejection():
    dt = 5e-6
    baseline = -70.
    t = np.arange(0, 3.0, dt)
    v = np.ones_like(t) * baseline
    i = np.ones_like(t) * 0.
    start = 1.
    end = 2.
    start_index = int(start / dt)
    end_index = int(end / dt)
    actual_tau = 0.02
    A = 10.

    v[start_index:end_index] = (baseline - A) + A * np.exp(-(t[start_index:end_index] - t[start_index]) / actual_tau)

    noise_level = 5.
    v += np.random.normal(scale=noise_level, size=len(v))

    tau = subf.time_constant(t, v, i, start=start, end=end)
    assert np.isnan(tau)


def test_time_constant_noise_acceptance():
    dt = 5e-6
    baseline = -70.
    t = np.arange(0, 3.0, dt)
    v = np.ones_like(t) * baseline
    i = np.ones_like(t) * 0.
    start = 1.
    end = 2.
    start_index = int(start / dt)
    end_index = int(end / dt)
    actual_tau = 0.02
    A = 10.

    v[start_index:end_index] = (baseline - A) + A * np.exp(-(t[start_index:end_index] - t[start_index]) / actual_tau)

    noise_level = 0.1
    np.random.seed(101)
    v += np.random.normal(scale=noise_level, size=len(v))

    tau = subf.time_constant(t, v, i, start=start, end=end)
    assert np.isclose(actual_tau, tau, rtol=1e-3)
