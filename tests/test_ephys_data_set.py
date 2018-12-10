import pandas as pd
import pytest
from ipfx.stimulus import StimulusOntology
from ipfx.ephys_data_set import EphysDataSet


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
    }

    df = pd.DataFrame(d)

    stimulus_names = ['EXTPBREAKN']
    default_ontology = StimulusOntology()
    ds = EphysDataSet(default_ontology)
    ds.sweep_table = df
    sweeps = ds.filtered_sweep_table(stimuli=stimulus_names)

    assert len(sweeps) == 2
