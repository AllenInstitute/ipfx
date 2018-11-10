import ipfx.subthresh_features as subf
import numpy as np


def test_input_resistance():
    t = np.arange(0, 1.0, 5e-6)
    v1 = np.ones_like(t) * -5.
    v2 = np.ones_like(t) * -10.
    i1 = np.ones_like(t) * -50.
    i2 = np.ones_like(t) * -100.

    ri = subf.input_resistance([t,t], [i1, i2], [v1,v2], 0, t[-1])

    assert np.allclose(ri, 100.)
