import pytest
import zipfile
import numpy as np
import io
import os
import sys
import urllib2
import shutil

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__),'data')


def load_array_from_zip_file(zip_file, file_name):
    with zipfile.ZipFile(os.path.join(TEST_DATA_PATH, zip_file), 'r') as f:
        sdata = f.read(file_name)
        data = np.genfromtxt(io.BytesIO(sdata))
    return data


@pytest.fixture()
def spike_test_pair():
    return load_array_from_zip_file("spike_test_pair.txt.zip", "spike_test_pair.txt")


@pytest.fixture()
def spike_test_var_dt():
    return load_array_from_zip_file("spike_test_var_dt.txt.zip", "spike_test_var_dt.txt")


@pytest.fixture()
def spike_test_high_init_dvdt():
    return load_array_from_zip_file("spike_test_high_init_dvdt.txt.zip", "spike_test_high_init_dvdt.txt")


@pytest.fixture()
def NWB_file(request):

    BASE_URL = "https://www.byte-physics.de/Downloads/allensdk-test-data/"

    nwb_file_name = request.param
    nwb_file_full_path = os.path.join(TEST_DATA_PATH, nwb_file_name)

    if not os.path.exists(nwb_file_full_path):
        response = urllib2.urlopen(BASE_URL + nwb_file_name)
        with open(nwb_file_full_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    return nwb_file_full_path


def pytest_addoption(parser):
    parser.addoption("--do-x-nwb-tests",
                     action="store_true",
                     default=False,
                     help="run file regression tests for conversion to NWBv2")


collect_ignore = []
if sys.version_info[0] < 3:
    collect_ignore.append("test_x_nwb.py")
    collect_ignore.append("test_x_nwb_helper.py")

