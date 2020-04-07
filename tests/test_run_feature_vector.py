from builtins import str
import os
import numpy as np
from ipfx.bin.run_feature_vector_extraction import run_feature_vector_extraction
from ipfx.bin.run_feature_collection import run_feature_collection
import pandas as pd
import pytest
from dictdiffer import diff

path_to_current_file = os.path.realpath(__file__)
current_directory = os.path.split(path_to_current_file)[0]

TEST_OUTPUT_DIR = os.path.join(current_directory, "data", "feature_vector")

nwb2_file1 = os.path.join(current_directory, "data", "Vip-IRES-Cre;Ai14(IVSCC)-226110.03.01.nwb")
nwb2_file2 = os.path.join(current_directory, "data", "Vip-IRES-Cre;Ai14(IVSCC)-236654.04.02.nwb")
test_nwb2_files = dict({500844783: nwb2_file1, 509604672: nwb2_file2})

def test_feature_vector_extraction(tmpdir_factory):

    temp_output_dir = str(tmpdir_factory.mktemp("feature_vector"))
    test_output_dir = TEST_OUTPUT_DIR

    features = [
        "first_ap_v",
        "first_ap_dv",
        "isi_shape",
        "psth",
        "inst_freq",
        "spiking_width",
        "spiking_peak_v",
        "spiking_fast_trough_v",
        "spiking_threshold_v",
        "spiking_upstroke_downstroke_ratio",
        "step_subthresh",
        "subthresh_norm",
        "subthresh_depol_norm",
        ]

    run_feature_vector_extraction(ids=[500844783, 509604672],
                                  output_dir=temp_output_dir,
                                  data_source="filesystem",
                                  output_code="TEMP",
                                  project=None,
                                  output_file_type="npy",
                                  sweep_qc_option="none",
                                  include_failed_cells=True,
                                  run_parallel=False,
                                  ap_window_length=0.003,
                                  file_list=test_nwb2_files
                                  )

    for feature in features:
        test_data = np.load(os.path.join(test_output_dir, "fv_{:s}_TEMP.npy".format(feature)))
        temp_data = np.load(os.path.join(temp_output_dir, "fv_{:s}_TEMP.npy".format(feature)))

        assert np.allclose(test_data, temp_data)


def test_feature_collection(tmpdir_factory):

    temp_output_dir = str(tmpdir_factory.mktemp("feature_vector"))
    test_output_dir = TEST_OUTPUT_DIR

    temp_output_file = os.path.join(temp_output_dir, "features_T301.csv")
    test_output_file = os.path.join(test_output_dir, "features_T301.csv")

    run_feature_collection(ids=[500844783, 509604672],
                           output_file=temp_output_file,
                           data_source="filesystem",
                           run_parallel=False,
                           file_list=test_nwb2_files)

    test_table = pd.read_csv(test_output_file, sep=",").to_dict()
    temp_table = pd.read_csv(temp_output_file, sep=",").to_dict()

    output_diff = list(diff(test_table, temp_table, tolerance=0.001))

    assert len(output_diff) == 0
