import ipfx.stim_features as st
import numpy as np


def test_get_stim_characteristics():

    i = [0,1,1,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t, test_pulse=False)
    assert start_time == 1
    assert np.isclose(duration, 2)
    assert np.isclose(amplitude,1)
    assert start_idx == 1
    assert end_idx == 2

    i = [0,0,1,1,0,0,3,3,3,3,0,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == 6
    assert np.isclose(duration,4)
    assert np.isclose(amplitude, 3)
    assert start_idx == 6
    assert end_idx == 9

    i = [0,0,-1,-1,0,0,3,3,3,3,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == 6
    assert np.isclose(duration,4)
    assert np.isclose(amplitude, 3)
    assert start_idx == 6
    assert end_idx == 9

    i = [0,0,-1,-1,0,0,-3,-3,-3,-3,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == 6
    assert np.isclose(duration,4)
    assert np.isclose(amplitude, -3)
    assert start_idx == 6
    assert end_idx == 9

    i = [0,-2,-1,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t, test_pulse=False)
    assert start_time == 1
    assert np.isclose(duration,2)
    assert np.isclose(amplitude, -2)
    assert start_idx == 1
    assert end_idx == 2

    i = [0,1,1,0,1,1,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == 4
    assert np.isclose(duration,2)
    assert np.isclose(amplitude, 1)
    assert start_idx == 4
    assert end_idx == 5

    i = [0,1,1,0,0,0,0,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == None
    assert duration ==None
    assert np.isclose(amplitude, 0)
    assert start_idx == None
    assert end_idx == None

    i = [0,11,11,11,11,11,0,0,0,0,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t,test_pulse=False)
    assert start_time == 1
    assert np.isclose(duration,5)
    assert np.isclose(amplitude, 11)
    assert start_idx == 1
    assert end_idx == 5

    i = [0,0,1,1,0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0,0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time == 6
    assert np.isclose(duration,7)
    assert np.isclose(amplitude, 0.7)
    assert start_idx == 6
    assert end_idx == 12


def test_find_stim_interval():
    a = [0,1,0,1,0,1,0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 2

    a = [0,1,1,0,0,1,1,0,0,1,1,0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 4

    a = [0,1,1,0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == None

    a = [0,1,1,0,0,1,1,0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 4

    a = [0,0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == None
