import allensdk.ipfx.qc_features as qcf
import numpy as np

def test_measure_blowout():
    a = [ 0,0,1,1 ]
    b = qcf.measure_blowout(a, 0)
    assert b == 500.0

    b = qcf.measure_blowout(a, 2)
    assert b == 1000.0

def test_measure_electrode_0():
    a = [1,1,1,1]
    b = qcf.measure_electrode_0(a, 1)
    assert np.isnan(b)

    b = qcf.measure_electrode_0(a, 1000)
    assert b == 1e12

def test_measure_seal():
    i = [0,0,0,0,1,1,1,0]
    v = [0,0,0,0,1,1,1,0]
    b = qcf.measure_seal(v, i, 2000.0)
    assert np.allclose([b],[1e-9])


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

