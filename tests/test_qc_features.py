from builtins import zip
import ipfx.qc_features as qcf
import numpy as np
import pytest

def test_measure_blowout():
    a = np.array([0, 0, 1, 1])
    b = qcf.measure_blowout(a, 0)
    assert b == 0.5

    b = qcf.measure_blowout(a, 2)
    assert b == 1.0


def test_measure_electrode_0():
    a = np.array([1, 1, 1, 1])
    b = qcf.measure_electrode_0(a, 1)
    assert b is None

    b = qcf.measure_electrode_0(a, 1000)
    assert b == 1


def test_measure_seal():
    i = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])
    v = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0])
    t = np.arange(len(v)) * 1E-3
    b = qcf.measure_seal(v, i, t)
    assert np.allclose([b], [1.0])


def test_measure_input_resistance():

    ir = 50.0
    dt = 1E-4
    time_range = [0, 0.5]
    time = np.arange(time_range[0], time_range[1], dt)
    current = np.zeros(time.shape)

    pulse_intervals = [(0.1, 0.2), (0.24, 0.3), (0.31, 0.32)]
    current_magnitudes = [1, 2, 3]

    for pulse_interval, current_magnitude in zip(pulse_intervals, current_magnitudes):

        ix = np.where((time > pulse_interval[0]) & (time < pulse_interval[1]))
        current[ix] = current_magnitude

    voltage = ir*current

    ir_tested = qcf.get_r_from_stable_pulse_response(voltage, current, time)
    assert np.isclose(ir_tested, ir)


def test_get_square_pulse_idx():

        v = [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0]
        up_idx = [6,11,15]
        down_idx = [7,12,17]

        assert up_idx,down_idx == qcf.get_square_pulse_idx(v)


def test_truncated_pulse():

    v = [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2]

    with pytest.raises(AssertionError, match="Truncated square pulse"):
        qcf.get_square_pulse_idx(v)


def test_negative_pulse():

    v = [0, 0, 1, 1, 0, 0, -2, -2, 0, 0, 0, 0, 2, 2, 2, 0, 0]

    with pytest.raises(AssertionError, match="Negative square pulse"):
        qcf.get_square_pulse_idx(v)
