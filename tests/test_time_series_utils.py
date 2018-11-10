import pytest
import numpy as np

import ipfx.time_series_utils as tsu


def test_find_time_out_of_bounds():
    t = np.array([0, 1, 2])
    t_0 = 4

    with pytest.raises(AssertionError):
        tsu.find_time_index(t, t_0)


def test_dvdt_no_filter():
    t = np.array([0, 1, 2, 3])
    v = np.array([1, 1, 1, 1])

    assert np.allclose(tsu.calculate_dvdt(v, t), np.diff(v) / np.diff(t))


def test_fixed_dt():
    t = [0, 1, 2, 3]
    assert tsu.has_fixed_dt(t) == True

    # Change the first time point to make time steps inconsistent
    t[0] -= 3.
    assert tsu.has_fixed_dt(t) == False

