import ipfx.epochs as ep
import pytest
import numpy as np

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

                             #   zero array
                             (
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 4,
                                 None
                             ),

                         ]
                         )
def test_get_experiment_epoch(i, sampling_rate, expt_epoch):
    assert expt_epoch == ep.get_experiment_epoch(i, sampling_rate)


@pytest.mark.parametrize('i,'
                         'sampling_rate,'
                         'test_epoch',
                         [
                             #   test pulse with square stim
                             (
                                 [0, 0, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0],
                                 4,
                                 (0, 5)
                             ),

                             #   missing test pulse
                             (
                                 [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0],
                                 4,
                                 None
                             ),

                             #   zero array
                             (
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 4,
                                 None
                             ),

                         ]
                         )
def test_get_test_epoch(i, sampling_rate, test_epoch):
    assert test_epoch == ep.get_test_epoch(i, sampling_rate)


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
                             #   zero array
                             (
                                 [0, 0, 0, 0, 0, 0, 0],
                                 None
                             ),

                         ]
                         )
def test_get_stim_epoch(i, stim_epoch):
    assert stim_epoch == ep.get_stim_epoch(i)


@pytest.mark.parametrize('response,'
                         'recording_epoch',
                         [
                             (
                                 [0, 0, 1, 1.5, 0, 0, 2, 3, 4, 1, np.nan, np.nan],
                                 (0, 9)
                             ),

                             #   zero array
                             (
                                 [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                 (0, 0)
                             ),

                         ]
                         )
def test_get_recording_epoch(response, recording_epoch):
    assert recording_epoch == ep.get_recording_epoch(response)


@pytest.mark.parametrize('response,'
                         'sweep_epoch',
                         [
                             (
                                 [0, 0, 1, 1.5, 0, 0, 2, 3, 4, 1, 0, 0],
                                 (0, 11)
                             ),

                             #   zero array
                             (
                                 [0, 0, 0, 0, 0, 0],
                                 (0, 5)
                             ),

                         ]
                         )
def test_get_sweep_epoch(response, sweep_epoch):
    assert sweep_epoch == ep.get_sweep_epoch(response)

