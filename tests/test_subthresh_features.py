import ipfx.subthresh_features as subf
import numpy as np


def test_input_resistance():
    # input_resistance now measures the deflection relative to a pre-stimulus
    # baseline, so the sweeps need a real baseline period before `start` and a
    # (gently-settling, to avoid transient rejection) step during the stimulus.
    dt = 5e-6
    t = np.arange(0, 1.0, dt)
    start = 0.3
    end = 0.8
    ramp_dur = 0.1

    def make_sweep(deflection, current):
        ramp = np.clip((t - start) / ramp_dur, 0.0, 1.0)
        v = np.where(t < start, 0.0, deflection * ramp)
        i = np.ones_like(t) * current
        return v, i

    # 100 MOhm: 0.1 mV/pA -> -50 pA gives -5 mV, -100 pA gives -10 mV
    v1, i1 = make_sweep(-5., -50.)
    v2, i2 = make_sweep(-10., -100.)

    ri = subf.input_resistance([t, t], [i1, i2], [v1, v2], start, end)

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
