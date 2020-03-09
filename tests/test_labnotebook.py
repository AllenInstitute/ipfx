from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import os
import pytest
import h5py
import numpy as np
from ipfx.dataset.labnotebook import LabNotebookReaderIgorNwb
from tests.helpers_for_tests import compare_dicts
from allensdk.api.queries.cell_types_api import CellTypesApi

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_labnotebook_reader(NWB_file):

    reader = LabNotebookReaderIgorNwb(NWB_file)
    assert isinstance(reader, LabNotebookReaderIgorNwb)

    assert reader.get_numerical_keys()[0][1] == 'TimeStamp'
    np.testing.assert_almost_equal(reader.get_numerical_values()[0][1][0], 3617183992.147)

    assert reader.get_textual_keys()[0][1] == 'TimeStamp'
    np.testing.assert_almost_equal(float(reader.get_textual_values()[0][1][0]), 3617200000.0)

    sweep_num = 0
    expected = {
        "Stim Wave Name": "EXTPSMOKET180424_DA_0",
        "Scale Factor": 0.5,
        "Set Sweep Count": 0.0
    }
    obtained = {}
    for k,v in expected.items():
        obtained[k] = reader.get_value(k, sweep_num, None)
    assert expected == obtained