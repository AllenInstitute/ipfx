import pandas as pd
import numpy as np
import pytest

from ipfx.stimulus import StimulusOntology
from ipfx.ephys_data_set import EphysDataSet

from helpers import compare_dicts


def get_sweep_table_dict():
    return {'bridge_balance_mohm': [15.6288156509, np.nan, np.nan],
            'clamp_mode': ['CurrentClamp', 'VoltageClamp', 'VoltageClamp'],
            'leak_pa': [-6.20449542999, np.nan, np.nan],
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


def test_get_sweep_info():

    d = get_sweep_table_dict()
    expected = {}
    for k in d:
        expected[k] = d[k][1]

    ds = get_dataset()
    actual = ds.get_sweep_info_by_sweep_number(5)
    compare_dicts(expected, actual)


def test_sweep_raises():

    with pytest.raises(NotImplementedError):
        ds = get_dataset()
        ds.sweep(5)


def test_set_sweep_raises_int():

    with pytest.raises(NotImplementedError):
        ds = get_dataset()
        ds.sweep_set(5)


def test_set_sweep_raises_list():

    with pytest.raises(NotImplementedError):
        ds = get_dataset()
        ds.sweep_set([5, 6])


def test_aligned_sweeps_raises():
    with pytest.raises(NotImplementedError):
        ds = get_dataset()
        ds.aligned_sweeps([5], 0.0)


def test_extract_sweep_meta_data_raises():
    with pytest.raises(NotImplementedError):
        ds = get_dataset()
        ds.extract_sweep_meta_data()
