import ipfx.stim_features as st
import numpy as np
import pytest


test_params = [
    (
        [0, 11, 11, 11, 11, 11, 0, 0, 0, 0, 0],  # i
        False,                                   # test_pulse
        (1, 5, 11, 1, 5)                         # stim_characteristics
     ),
    (
        [0, 0, 1, 1, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0, 0],
        True,
        (6, 7, 0.7, 6, 12)
     ),
    (
        [0, 1, 1, 0, 1, 1, 0],
        True,
        (4, 2, 1, 4, 5)
    ),
    (
        [0, -2, -1, 0],
        False,
        (1, 2, -2, 1, 2)
    ),
    (
        [0, 0, -1, -1, 0, 0, -3, -3, -3, -3, 0],
        True,
        (6, 4, -3, 6, 9)
    ),
    (
        [0, 0, -1, -1, 0, 0, 3, 3, 3, 3, 0],
        True,
        (6, 4, 3, 6, 9)
    ),
    (
        [0, 0, 1, 1, 0, 0, 3, 3, 3, 3, 0, 0],
        True,
        (6, 4, 3, 6, 9)
    ),
    (
        [0, 1, 1, 0],
        False,
        (1, 2, 1, 1, 2)
    )
]


@pytest.mark.parametrize('i, test_pulse, stim_characteristics', test_params)
def test_get_stimuli_characteristics(i, test_pulse, stim_characteristics):
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t, test_pulse=test_pulse)
    assert start_time == stim_characteristics[0]
    assert np.isclose(duration, stim_characteristics[1])
    assert np.isclose(amplitude, stim_characteristics[2])
    assert start_idx == stim_characteristics[3]
    assert end_idx == stim_characteristics[4]


def test_none_get_stimuli_characteristics():

    i = [0, 1, 1, 0, 0, 0, 0, 0]
    t = np.arange(len(i))
    start_time, duration, amplitude, start_idx, end_idx = st.get_stim_characteristics(i, t)
    assert start_time is None
    assert duration is None
    assert np.isclose(amplitude, 0)
    assert start_idx is None
    assert end_idx is None


def test_find_stim_interval():
    a = [0, 1, 0, 1, 0, 1, 0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 2

    a = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 4

    a = [0, 1, 1, 0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == None  # noqa: E711

    a = [0, 1, 1, 0, 0, 1, 1, 0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == 4

    a = [0, 0]
    interval = st.find_stim_interval(0, a, hz=1)
    assert interval == None  # noqa: E711
