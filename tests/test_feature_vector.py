from __future__ import print_function
import numpy as np
from ipfx.aibs_data_set import AibsDataSet
import ipfx.data_set_features as dsf
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.feature_vectors as fv
import pytest
import os


TEST_OUTPUT_DIR = "/allen/aibs/informatics/module_test_data/ipfx/test_feature_vector"


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

    temp_data = fv.subthresh_depol_norm(sweeps, features, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "step_subthresh.npy"))

    assert np.array_equal(test_data, temp_data)


def test_subthresh_depol_norm(feature_vector_input):

    sweeps, features, start, end = feature_vector_input

    temp_data = fv.subthresh_depol_norm(sweeps, features, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "subthresh_depol_norm.npy"))

    assert np.array_equal(test_data, temp_data)

