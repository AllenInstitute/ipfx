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
