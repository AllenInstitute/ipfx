import os
import pytest
import urllib2
import shutil

from ipfx.stimulus import StimulusOntology
from ipfx.hbg_dataset import HBGDataSet

from helpers import compare_dicts


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


@pytest.fixture()
def fetch_DAT_NWB_file():
    output_filepath = 'H18.28.015.11.14.nwb'
    if not os.path.exists(output_filepath):

        BASE_URL = "https://www.byte-physics.de/Downloads/allensdk-test-data/"

        response = urllib2.urlopen(BASE_URL + output_filepath)
        with open(output_filepath, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def test_main_abf(ontology):
    dataset = HBGDataSet(nwb_file=os.path.join(os.path.dirname(__file__), 'data',
                                               '2018_03_20_0005.nwb'), ontology=ontology)

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


def test_main_dat(ontology, fetch_DAT_NWB_file):
    dataset = HBGDataSet(nwb_file='H18.28.015.11.14.nwb', ontology=ontology)

    expected = {'stimulus_units': {0: 'V'},
                'clamp_mode': {0: 'VoltageClamp'},
                'sweep_number': {0: 10101},
                'leak_pa': {0: None},
                'stimulus_code_ext': {0: None},
                'stimulus_scale_factor': {0: 5000000.0},
                'stimulus_code': {0: u'extpinbath'},
                'stimulus_name': {0: u'extpinbath stimulus'},
                'bridge_balance_mohm': {0: None}
                }

    # only compare one sweep
    sweep_table = dataset.filtered_sweep_table(sweep_number=10101)

    compare_dicts(expected, sweep_table.to_dict())
