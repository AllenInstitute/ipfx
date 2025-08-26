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
nwb2_nwb_schema_2_9_0_file1 = os.path.join(current_directory, "data", "Vip-IRES-Cre;Ai14(IVSCC)-226110.03.01_ITC18USB_Dev_0-nwb-schema-2.9.0.nwb")
nwb2_nwb_schema_2_9_0_file2 = os.path.join(current_directory, "data", "Vip-IRES-Cre;Ai14(IVSCC)-236654.04.02_ITC18USB_Dev_0-nwb-schema-2.9.0.nwb")

testdata = [dict({500844783: nwb2_file1,
                  509604672: nwb2_file2}),
            dict({500844783: nwb2_nwb_schema_2_9_0_file1,
                  509604672: nwb2_nwb_schema_2_9_0_file2})
           ]

@pytest.mark.parametrize("ids_and_files", testdata, ids = ["default", "schema-2.9.0"])
def test_feature_vector_extraction(ids_and_files, tmpdir_factory):

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

    run_feature_vector_extraction(ids=ids_and_files.keys(),
                                  output_dir=temp_output_dir,
                                  data_source="filesystem",
                                  output_code="TEMP",
                                  project=None,
                                  output_file_type="npy",
                                  sweep_qc_option="none",
                                  include_failed_cells=True,
                                  run_parallel=False,
                                  ap_window_length=0.003,
                                  file_list=ids_and_files
                                  )

    for feature in features:
        test_data = np.load(os.path.join(test_output_dir, "fv_{:s}_TEMP.npy".format(feature)))
        temp_data = np.load(os.path.join(temp_output_dir, "fv_{:s}_TEMP.npy".format(feature)))

        assert np.allclose(test_data, temp_data)


@pytest.mark.parametrize("ids_and_files", testdata, ids = ["default", "schema-2.9.0"])
def test_feature_collection(ids_and_files, tmpdir_factory):

    temp_output_dir = str(tmpdir_factory.mktemp("feature_vector"))
    test_output_dir = TEST_OUTPUT_DIR

    temp_output_file = os.path.join(temp_output_dir, "features_T301.csv")
    test_output_file = os.path.join(test_output_dir, "features_T301.csv")

    run_feature_collection(ids=ids_and_files.keys(),
                           output_file=temp_output_file,
                           data_source="filesystem",
                           run_parallel=False,
                           file_list=ids_and_files)

    test_table = pd.read_csv(test_output_file, sep=",").to_dict()
    temp_table = pd.read_csv(temp_output_file, sep=",").to_dict()

    output_diff = list(diff(test_table, temp_table, tolerance=0.001))

    assert len(output_diff) == 0
