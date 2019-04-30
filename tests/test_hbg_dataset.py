import pytest

from ipfx.stimulus import StimulusOntology
from ipfx.hbg_dataset import HBGDataSet

from helpers_for_tests import compare_dicts


@pytest.fixture()
def ontology():
    return StimulusOntology([[('name', 'ramp stimulus'), ('code', 'RAMP1')],
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


@pytest.mark.parametrize('ontology, NWB_file', [(None, '2018_03_20_0005.nwb')], indirect=True)
def test_main_abf(ontology, NWB_file):

    dataset = HBGDataSet(nwb_file=NWB_file, ontology=ontology)

    expected = {'stimulus_units': {0: 'A'},
                'clamp_mode': {0: 'CurrentClamp'},
                'sweep_number': {0: 0},
                'leak_pa': {0: None},
                'stimulus_code_ext': {0: None},
                'stimulus_scale_factor': {0: 1.0},
                'stimulus_code': {0: u'RAMP1'},
                'stimulus_name': {0: u'ramp stimulus'},
                'bridge_balance_mohm': {0: None}
                }

    compare_dicts(expected, dataset.sweep_table.to_dict())


@pytest.mark.parametrize('ontology, NWB_file', [(None, 'H18.28.015.11.14.nwb')], indirect=True)
def test_main_dat(ontology, NWB_file):

    dataset = HBGDataSet(nwb_file=NWB_file, ontology=ontology)

    expected = {'stimulus_units': 'V',
                'clamp_mode': 'VoltageClamp',
                'sweep_number': 10101,
                'leak_pa': None,
                'stimulus_code_ext': None,
                'stimulus_scale_factor': 5000000.0,
                'stimulus_code': u'extpinbath',
                'stimulus_name': u'extpinbath stimulus',
                'bridge_balance_mohm': None
                }

    # only compare one sweep
    sweep_record = dataset.get_sweep_record(10101)
    compare_dicts(expected, sweep_record)


@pytest.mark.parametrize('ontology, NWB_file', [(None, 'H18.28.015.11.14.nwb')], indirect=True)
def test_get_clamp_mode(ontology, NWB_file):

    dataset = HBGDataSet(nwb_file=NWB_file, ontology=ontology)
    assert dataset.get_clamp_mode(10101) == dataset.VOLTAGE_CLAMP


@pytest.mark.parametrize('ontology, NWB_file', [(None, 'H18.28.015.11.14.nwb')], indirect=True)
def test_get_stimulus_units(ontology, NWB_file):

    dataset = HBGDataSet(nwb_file=NWB_file, ontology=ontology)
    assert dataset.get_stimulus_units(10101) == "V"


@pytest.mark.parametrize('ontology, NWB_file', [(None, 'H18.28.015.11.14.nwb')], indirect=True)
def test_get_stimulus_code(ontology, NWB_file):

    dataset = HBGDataSet(nwb_file=NWB_file, ontology=ontology)
    assert dataset.get_stimulus_code(10101) == u'extpinbath'


