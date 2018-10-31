import pytest
import zipfile
import numpy as np
import io
import os
import sys

PATH = os.path.join(os.path.dirname(__file__))

def load_array_from_zip_file(zip_file, file_name):
    with zipfile.ZipFile(os.path.join(PATH, zip_file), 'r') as f:
        sdata = f.read(file_name)
        data = np.genfromtxt(io.BytesIO(sdata))
    return data


@pytest.fixture()
def spike_test_pair():
    return load_array_from_zip_file("data/spike_test_pair.txt.zip", "spike_test_pair.txt")

@pytest.fixture()
def spike_test_var_dt():
    return load_array_from_zip_file("data/spike_test_var_dt.txt.zip", "spike_test_var_dt.txt")

@pytest.fixture()
def spike_test_high_init_dvdt():
    return load_array_from_zip_file("data/spike_test_high_init_dvdt.txt.zip", "spike_test_high_init_dvdt.txt")

def pytest_addoption(parser):
    parser.addoption("--do-x-nwb-tests",
                     action="store_true",
                     default=False,
                     help="run file regression tests for conversion to NWBv2")


collect_ignore = []
if sys.version_info[0] < 3:
    collect_ignore.append("test_x_nwb.py")
    collect_ignore.append("test_x_nwb_helper.py")
