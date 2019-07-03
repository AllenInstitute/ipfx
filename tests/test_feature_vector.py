from __future__ import print_function
import numpy as np
from ipfx.aibs_data_set import AibsDataSet
import ipfx.data_set_features as dsf
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.feature_vectors as fv
from ipfx.sweep import Sweep, SweepSet
import pytest
import os


TEST_OUTPUT_DIR = "/allen/aibs/informatics/module_test_data/ipfx/test_feature_vector"

random_sweep_list = []
sampling_rate = 1
clamp_mode = "CurrentClamp"

@pytest.fixture
def feature_vector_input():

    TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

    nwb_file = os.path.join(TEST_DATA_PATH, "Pvalb-IRES-Cre;Ai14-415796.02.01.01.nwb")
    data_set = AibsDataSet(nwb_file= nwb_file)
    ontology = data_set.ontology

    lsq_sweep_numbers = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
                                                      stimuli=ontology.long_square_names).sweep_number.sort_values().values

    lsq_sweeps = data_set.sweep_set(lsq_sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(lsq_sweeps.sweeps[0].i,
                                                               lsq_sweeps.sweeps[0].t)

    lsq_end = lsq_start + lsq_dur
    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                  start=lsq_start,
                                                  end=lsq_end,
                                                  **dsf.detection_parameters(data_set.LONG_SQUARE))
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)

    lsq_features = lsq_an.analyze(lsq_sweeps)

    return lsq_sweeps, lsq_features, lsq_start, lsq_end


def test_isi_shape(feature_vector_input):

    sweeps, features, start, end = feature_vector_input

    temp_data = fv.isi_shape(sweeps, features)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "isi_shape.npy"))

    assert np.array_equal(test_data, temp_data)


def test_step_subthreshold(feature_vector_input):
    sweeps, features, start, end = feature_vector_input
    target_amps = [-90, -70, -50, -30, -10]

    subthresh_hyperpol_dict = fv.identify_subthreshold_hyperpol_with_amplitudes(features, sweeps)
    temp_data = fv.step_subthreshold(subthresh_hyperpol_dict, target_amps, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "step_subthresh.npy"))

    assert np.array_equal(test_data, temp_data)


def test_step_subthreshold_interpolation():
    target_amps = [-90, -70, -50, -30, -10]
    test_sweep_list = []
    test_amps = [-70, -30]
    t = np.arange(6)
    i = np.zeros_like(t)
    epochs = {"sweep": (0, 5), "test": None, "recording": None, "experiment": None, "stim": None}
    for a in test_amps:
        v = np.hstack([np.zeros(2), np.ones(2) * a, np.zeros(2)])
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        test_sweep_list.append(test_sweep)
    amp_sweep_dict = dict(zip(test_amps, test_sweep_list))
    output = fv.step_subthreshold(amp_sweep_dict, target_amps, start=2, end=4,
        extend_duration=1, subsample_interval=1)
    assert np.all(output[1:3] == -90)
    assert np.array_equal(output[4:8], test_sweep_list[0].v[1:-1])
    assert np.all(output[9:11] == -50)
    assert np.array_equal(output[12:16], test_sweep_list[1].v[1:-1])
    assert np.all(output[17:19] == -10)


def test_subthresh_depol_norm(feature_vector_input):

    sweeps, features, start, end = feature_vector_input

    temp_data = fv.subthresh_depol_norm(sweeps, features, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "subthresh_depol_norm.npy"))

    assert np.array_equal(test_data, temp_data)

