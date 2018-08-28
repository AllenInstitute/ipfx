import pandas as pd
import pytest
from allensdk.ipfx.stimulus import load_default_stimulus_ontology, StimulusOntology
from allensdk.ipfx.ephys_data_set import EphysDataSet


@pytest.fixture()
def ontology():
    return StimulusOntology([ [ ('name', 'long square'),
                                ('code', 'LS') ],
                              [ ('name', 'noise', 'noise 1'),
                                ('code', 'C1NS1') ],
                              [ ('name', 'noise', 'noise 2'),
                                ('code', 'C1NS2') ] ])


def test_load_default():
    load_default_stimulus_ontology()

def test_find(ontology):
    stims = ontology.find('C1NS1')

    stims = ontology.find('noise')
    assert len(stims) == 2

def test_find_one(ontology):
    stim = ontology.find_one('LS')

    assert stim.tags(tag_type='name')[0][-1] == 'long square'

    with pytest.raises(KeyError):
        stims = ontology.find_one('noise')



def test_has(ontology):
    assert ontology.stimulus_has_any_tags('C1NS1', ('noise',))
    assert ontology.stimulus_has_any_tags('C1NS1', ('noise','noise 2'))
    assert not ontology.stimulus_has_all_tags('C1NS1', ('noise','noise 2'))


def test_filtered_sweep_table():

    d = {'bridge_balance_mohm': [15.6288156509,None,None],
    'clamp_mode': ['CurrentClamp','VoltageClamp','VoltageClamp'],
    'leak_pa' : [-6.20449542999,None,None],
    'stimulus_code': ['C1LSFINEST150112', 'EXTPBREAKN141203', 'EXTPBREAKN141203'],
    'stimulus_code_ext': ['C1LSFINEST150112[0]','EXTPBREAKN141203[0]','EXTPBREAKN141203[0]'],
    'stimulus_name': ["Long Square","Test","Test"],
    'stimulus_scale_factor':[10.0,0.5,0.5],
    'stimulus_units': ['pA','mV','mV'],
    'sweep_number': [49,5,6],
    'truncated': [None,None,None]
    }

    df = pd.DataFrame(d)

    stimulus_names = ['EXTPBREAKN']
    ds = EphysDataSet()
    ds.sweep_table = df
    sweeps = ds.filtered_sweep_table(stimuli=stimulus_names)

    assert len(sweeps) == 2
