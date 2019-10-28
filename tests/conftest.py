import pytest
import zipfile
import numpy as np
import io
import os
import sys
from .helpers_for_tests import download_file
import ipfx.bin.lims_queries as lq


TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
TEST_DATA_PATH_INHOUSE = "/allen/aibs/informatics/module_test_data/ipfx/nwb"


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

    nwb_file_name = request.param
    nwb_file_full_path = os.path.join(TEST_DATA_PATH, nwb_file_name)

    if not os.path.exists(nwb_file_full_path):
        download_file(nwb_file_name, nwb_file_full_path)

    return nwb_file_full_path


@pytest.fixture()
def NWB_file_inhouse(request):

    nwb_file_name = request.param
    nwb_file_full_path = os.path.join(TEST_DATA_PATH_INHOUSE, nwb_file_name)

    return nwb_file_full_path


def pytest_addoption(parser):
    parser.addoption("--do-x-nwb-tests",
                     action="store_true",
                     default=False,
                     help="run file regression tests for conversion to NWBv2")


def pytest_configure(config):
    config.addinivalue_line("markers", "xnwbtest: mark test as part of NWB conversion set")



def pytest_collection_modifyitems(config, items):
    """
    A pytest magic function. This function is called post-collection and gives us a hook for modifying the
    collected items.

    """

    skip_requires_lims = pytest.mark.skipif(
        not lq.able_to_connect_to_lims(),
        reason='This test requires connection to lims'
    )
    if config.getoption("--do-x-nwb-tests"):
        return
    skip_xnwb = pytest.mark.skip(reason="need --do-x-nwb-tests to run")

    skip_internal = pytest.mark.skipif(
        os.getenv('TEST_INHOUSE') != 'true',
        reason='depend on resources internal to the Allen Institute.'
    )

    for item in items:
        if 'requires_lims' in item.keywords:
            item.add_marker(skip_requires_lims)

        if "xnwbtest" in item.keywords:
           item.add_marker(skip_xnwb)

        if "requires_inhouse_data" in item.keywords:
            item.add_marker(skip_internal)
