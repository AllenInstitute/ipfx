from __future__ import print_function
from __future__ import absolute_import
from builtins import range
import os
import pytest
import h5py
import numpy as np
from ipfx.dataset.labnotebook import LabNotebookReaderIgorNwb

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_labnotebook_reader(NWB_file):

    reader = LabNotebookReaderIgorNwb(NWB_file)
    assert isinstance(reader, LabNotebookReaderIgorNwb)

    sweep_num = 0
    expected = {
        "Stim Wave Name": "EXTPSMOKET180424_DA_0",
        "Scale Factor": 0.5,
        "Set Sweep Count": 0.0,
        "Stimset Acq Cycle ID":  7394437.0,
        "TimeStamp":  3617184191.612,
        "TimeStampSinceIgorEpochUTC":  3617209391.612,
        "EntrySourceType":  1.0,
        "TP Baseline Vm":  -167.69015502929688,
        "TP Baseline pA":  51.41988754272461,
        "Headstage Active":  1.0,
        "Stim Scale Factor" :  1.0,
        "Stim set length" :  46000.0,
        "DA unit" :  "mV",
        "DA Gain" :  20.0,
        "AD unit" :  "pA",
        "AD Gain" :  0.0005000000237487257
    }
    obtained = {}
    for k,v in expected.items():
        obtained[k] = reader.get_value(k, sweep_num, None)
    assert expected == obtained