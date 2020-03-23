from __future__ import absolute_import
import pytest
import os

import pynwb
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.hbg_nwb_data import HBGNWB2Data
import allensdk.core.json_utilities as ju
from dictdiffer import diff
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

    ontology =  StimulusOntology([[('name', 'ramp stimulus'), ('code', 'RAMP1')],
                             [('name', 'extpinbath stimulus'), ('code', 'extpinbath')],
                             [('name', 'extpbreakn stimulus'), ('code', 'extpbreakn')],
                             [('name', 'Long square stimulus'), ('code', 'Long square')],
                             [('name', 'Short square stimulus'), ('code', 'Short square')],
                             [('name', 'Rheobase stimulus'), ('code', 'Rheobase')],
                             [('name', 'Ramp stimulus'), ('code', 'Ramp')],
                             [('name', 'Capacitance stimulus'), ('code', 'Capacitance')],
                             [('name', 'Chirp stimulus'), ('code', 'Chirp')],
                             [('name', 'extpexpend stimulus'), ('code', 'extpexpend')]
                             ])

    return HBGNWB2Data(nwb_file=tmp_nwb_path, ontology=ontology)

def test_create_hbg(hbg_nwb_data):
    assert isinstance(hbg_nwb_data,HBGNWB2Data)

def test_main_dat(hbg_nwb_data):

    expected = {
        'sweep_number': 4,
        'stimulus_units': 'amperes',
        'bridge_balance_mohm': 500.0,
        'leak_pa': 100.0,
        'stimulus_scale_factor': 32.0,
        'stimulus_code': 'STIMULUS_CODE',
        'stimulus_code_ext': 'STIMULUS_CODE',
        'stimulus_name': 'Unknown'}

    obtained = hbg_nwb_data.get_sweep_record(sweep_num=4)
    assert expected == obtained

# @pytest.mark.parametrize('ontology, NWB_file', [(None, 'H20.28.008.11.05-10.nwb')], indirect=True)
# def test_get_clamp_mode(ontology, NWB_file):

#     dataset = HBGNWB2Data(nwb_file=NWB_file, ontology=ontology())
#     assert dataset.get_clamp_mode(10101) == dataset.VOLTAGE_CLAMP



