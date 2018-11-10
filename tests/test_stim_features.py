import pytest
import ipfx.stim_features as st



def test_find_stim_start():
    a = [0,1,1,0]
    stim_start = st.find_stim_start(a)
    assert stim_start == 1

    a = [0,1,1,0,1,1,0]
    stim_start = st.find_stim_start(a, idx0=3)
    assert stim_start == 4

    a = [1,1,0,0,0,0,0]
    stim_start = st.find_stim_start(a)
    assert stim_start == 2

def test_find_stim_window():
    a = [0,1,1,0]
    stim_start, stim_dur = st.find_stim_window(a)
    assert stim_start == 1
    assert stim_dur == 2

    a = [0,1,1,0,1,1,0]
    stim_start, stim_dur = st.find_stim_window(a, idx0=3)
    assert stim_start == 4
    assert stim_dur == 2

    a = [1,1,0,0,0,0,0]
    stim_start, stim_dur = st.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 5

    a = [1,1,0,0,0,0,1]
    stim_start, stim_dur = st.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4

    a = [1,1,0,0,1,0,1]
    stim_start, stim_dur = st.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4

def test_find_stim_amplitude_and_duration():
    a = [0,1,1,0]
    amp, dur = st.find_stim_amplitude_and_duration(0, a, hz=1)
    assert amp == 1
    assert dur == 2

    a = [0,-2,-1,0]
    amp, dur = st.find_stim_amplitude_and_duration(0, a, hz=1)
    assert amp == -2
    assert dur == 2

    a = [0,1,1,0,1,1,0]
    amp, dur = st.find_stim_amplitude_and_duration(3, a, hz=1)
    assert amp == 1
    assert dur == 2

    a = [1,1,0,0,0,0,0]
    amp, dur = st.find_stim_amplitude_and_duration(0, a, hz=1)
    assert amp == 1
    assert dur == 2

    a = [1,1,0,0,0,0,1]
    amp, dur = st.find_stim_amplitude_and_duration(0, a, hz=1)
    assert amp == 1
    assert dur == 7

    a = [1,1,0,0,1,0,1]
    amp, dur = st.find_stim_amplitude_and_duration(0, a, hz=1)
    assert amp == 1
    assert dur == 7

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
