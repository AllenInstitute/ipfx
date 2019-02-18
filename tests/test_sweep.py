from ipfx.sweep import Sweep
import pytest
import numpy as np

@pytest.fixture()
def sweep():

    i = [0,0,1,1,0,0,0,2,2,2,2,2,0,0,0,0]
    v = [0,0,1,2,1,0,0,1,2,3,1,0,0,0,0,0]
    sampling_rate = 2
    dt = 1./sampling_rate
    t = np.arange(0,len(v))*dt

    return Sweep(t, v, i, sampling_rate=sampling_rate, clamp_mode="CurrentClamp")


def test_select_epoch(sweep):

    i_sweep = sweep.i
    v_sweep = sweep.v

    sweep.select_epoch("recording")
    assert sweep.i == [0,0,1,1,0,0,0,2,2,2,2]
    assert sweep.v == [0,0,1,2,1,0,0,1,2,3,1]

    sweep.select_epoch("sweep")
    assert sweep.i == i_sweep
    assert sweep.v == v_sweep


def test_set_time_zero_to_index(sweep):

    t0_idx = 7
    sweep.set_time_zero_to_index(t0_idx)

    assert np.isclose(sweep.t[t0_idx], 0.0)


