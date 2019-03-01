import ipfx.epochs as ep
import pytest


def test_find_stim_window():
    a = [0, 1, 1, 0]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 1
    assert stim_dur == 2

    a = [0, 1, 1, 0, 1, 1, 0]
    stim_start, stim_dur = ep.find_stim_window(a, idx0=3)
    assert stim_start == 4
    assert stim_dur == 2

    a = [1, 1, 0, 0, 0, 0, 0]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 5

    a = [1, 1, 0, 0, 0, 0, 1]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4

    a = [1, 1, 0, 0, 1, 0, 1]
    stim_start, stim_dur = ep.find_stim_window(a)
    assert stim_start == 2
    assert stim_dur == 4


@pytest.mark.parametrize('i,'
                         'sampling_rate,'
                         'expt_epoch',
                         [
                             #   test pulse with square stim
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
                                 4,
                                 (6, 14)
                             ),

                             #   test pulse with ramp stim
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
                                 4,
                                 (5, 16)
                             ),

                             #   test pulse with triple square
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                                 4,
                                 (5, 16)
                             ),

                             #   negative test pulse with short square stim
                             (
                                 [0, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0],
                                 4,
                                 (4, 9)
                             ),

                         ]
                         )
def test_get_experiment_epoch(i, sampling_rate, expt_epoch):
    assert expt_epoch == ep.get_experiment_epoch(i, sampling_rate)


@pytest.mark.parametrize('i,'
                         'stim_epoch',
                         [
                             #   test pulse with square stim
                             (
                                 [0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0],
                                 (6, 10)
                             ),

                             #   test pulse with ramp stim
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
                                 (7, 14)
                             ),

                             #   test pulse with triple square
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0],
                                 (7, 14)
                             ),

                             #   negative test pulse with short square stim
                             (
                                 [0, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0],
                                 (6, 7)
                             ),
                         ]
                         )
def test_get_stim_epoch(i, stim_epoch):
    assert stim_epoch == ep.get_stim_epoch(i)


def test_get_stim_epoch_error():

    i = [0, 0, 0, 0, 0, 0, 0]

    with pytest.raises(ValueError):
        ep.get_stim_epoch(i)
