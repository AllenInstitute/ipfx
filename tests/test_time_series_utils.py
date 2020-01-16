import pytest
import numpy as np

import ipfx.time_series_utils as tsu
from ipfx.error import FeatureError


def test_find_time_out_of_bounds():
    t = np.array([0, 1, 2])
    t_0 = 4

    with pytest.raises(FeatureError):
        tsu.find_time_index(t, t_0)


def test_dvdt_no_filter():
    t = np.array([0, 1, 2, 3])
    v = np.array([1, 1, 1, 1])

    assert np.allclose(tsu.calculate_dvdt(v, t), np.diff(v) / np.diff(t))


def test_fixed_dt():
    t = [0, 1, 2, 3]
    assert tsu.has_fixed_dt(t)

    # Change the first time point to make time steps inconsistent
    t[0] -= 3.
    assert not tsu.has_fixed_dt(t)

def test_flatnotnan():
    a = [1, 10, 12, 17, 13, 4, 8, np.nan, np.nan]

    assert np.all(tsu.flatnotnan(a) == [0, 1, 2, 3, 4, 5, 6])

