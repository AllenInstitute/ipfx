import pandas as pd
import pytest

from ipfx.stimulus import StimulusOntology
from ipfx.ephys_data_set import EphysDataSet


def get_sweep_table_dict():
    return {'bridge_balance_mohm': [15.6288156509, None, None],
            'clamp_mode': ['CurrentClamp', 'VoltageClamp', 'VoltageClamp'],
            'leak_pa': [-6.20449542999, None, None],
            'stimulus_code': ['C1LSFINEST150112', 'EXTPBREAKN141203', 'EXTPBREAKN141203'],
            'stimulus_code_ext': ['C1LSFINEST150112[0]', 'EXTPBREAKN141203[0]', 'EXTPBREAKN141203[0]'],
            'stimulus_name': ["Long Square", "Test", "Test"],
            'stimulus_scale_factor': [10.0, 0.5, 0.5],
            'stimulus_units': ['pA', 'mV', 'mV'],
            'sweep_number': [49, 5, 6]}


def get_dataset():

    d = get_sweep_table_dict()
    df = pd.DataFrame(d)
    default_ontology = StimulusOntology()
    dataset = EphysDataSet(default_ontology)
    dataset.sweep_table = df

    return dataset


def test_get_sweep_number_by_stimulus_name_invalid_sweep():

    with pytest.raises(IndexError):
        ds = get_dataset()
        ds.get_sweep_number_by_stimulus_names(['I_DONT_EXIST'])


def test_get_sweep_number_by_stimulus_name_works_1():
    ds = get_dataset()
    sweeps = ds.get_sweep_number_by_stimulus_names(['C1LSFINEST'])
    assert sweeps == 49


def test_get_sweep_number_by_stimulus_name_works_and_returns_only_the_last():
    ds = get_dataset()
    sweeps = ds.get_sweep_number_by_stimulus_names(['EXTPBREAKN'])
    assert sweeps == 6


def test_filtered_sweep_table_works():

    ds = get_dataset()
    sweeps = ds.filtered_sweep_table(stimuli=['EXTPBREAKN'])

    assert sweeps["sweep_number"].tolist() == [5, 6]
