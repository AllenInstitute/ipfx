import os
import numpy as np
from ipfx.bin.run_feature_vector_extraction import run_feature_vector_extraction
from ipfx.bin.run_feature_collection import run_feature_collection
import pandas as pd
import pytest
from dictdiffer import diff


TEST_OUTPUT_DIR = "/allen/aibs/informatics/module_test_data/ipfx/test_feature_vector"


@pytest.mark.requires_lims
@pytest.mark.regression
def test_feature_vector_extraction(tmpdir_factory):

    temp_output_dir = str(tmpdir_factory.mktemp("feature_vector"))
    test_output_dir = TEST_OUTPUT_DIR

    features = [
        "first_ap",
        "isi_shape",
        "spiking",
        "step_subthresh",
        "subthresh_norm",
        ]

    run_feature_vector_extraction(ids=[500844783, 509604672],
                                  output_dir=temp_output_dir)

    for feature in features:
        test_data = np.load(os.path.join(test_output_dir, "fv_{:s}_T301.npy".format(feature)))
        temp_data = np.load(os.path.join(temp_output_dir, "fv_{:s}_T301.npy".format(feature)))

        assert np.array_equal(test_data, temp_data)


@pytest.mark.requires_lims
@pytest.mark.regression
def test_feature_collection(tmpdir_factory):

    temp_output_dir = str(tmpdir_factory.mktemp("feature_vector"))
    test_output_dir = TEST_OUTPUT_DIR

    temp_output_file = os.path.join(temp_output_dir, "features_T301.csv")
    test_output_file = os.path.join(test_output_dir, "features_T301.csv")

    run_feature_collection(ids=[500844783, 509604672],
                           output_file=temp_output_file)

    test_table = pd.read_csv(test_output_file, sep=",").to_dict()
    temp_table = pd.read_csv(temp_output_file, sep=",").to_dict()

    output_diff = list(diff(test_table, temp_table, tolerance=0.001))

    assert len(output_diff) == 0


