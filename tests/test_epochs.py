import ipfx.epochs as ep


def test_find_stim_window():
    a = [0,1,1,0]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 1
    assert stim_dur == 2

    a = [0,1,1,0,1,1,0]
    stim_start, stim_dur = ep.find_stim_window(a, idx0=3)
    assert stim_start == 4
    assert stim_dur == 2

    a = [1,1,0,0,0,0,0]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 5

    a = [1,1,0,0,0,0,1]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4

    a = [1,1,0,0,1,0,1]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4

