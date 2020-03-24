import pytest

import pynwb
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.hbg_nwb_data import HBGNWBData
from tests.dataset.test_ephys_nwb_data import nwbfile_to_test

@pytest.fixture
def tmp_nwb_path(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test_nwb_data").join("test_hbg_data.nwb")
    return str(nwb)

@pytest.fixture
def hbg_nwb_data(tmp_nwb_path):

    nwbfile = nwbfile_to_test()
    print(tmp_nwb_path)

    with pynwb.NWBHDF5IO(path=tmp_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    ontology =  StimulusOntology(
        [[('name', 'expected name'), ('code', 'STIMULUS_CODE')],
         [('name', 'test name'), ('code', 'extpexpend')],
         ])

    return HBGNWBData(nwb_file=tmp_nwb_path, ontology=ontology)


def test_create_hbg(hbg_nwb_data):
    assert isinstance(hbg_nwb_data,HBGNWBData)


def test_get_sweep_metadata(hbg_nwb_data):

    expected = {
        'sweep_number': 4,
        'stimulus_units': 'amperes',
        'bridge_balance_mohm': 500.0,
        'leak_pa': 100.0,
        'stimulus_scale_factor': 32.0,
        'stimulus_code': 'STIMULUS_CODE',
        'stimulus_code_ext': 'STIMULUS_CODE',
        'stimulus_name': 'expected name',
        'clamp_mode': 'CurrentClamp',
    }

    obtained = hbg_nwb_data.get_sweep_metadata(sweep_number=4)
    assert expected == obtained

def  test_get_stimulus_code_ext(hbg_nwb_data):

    expected = 'STIMULUS_CODE'
    obtained = hbg_nwb_data.get_stimulus_code_ext(sweep_number=4)
    assert expected == obtained

