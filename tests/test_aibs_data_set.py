from __future__ import absolute_import
import pytest
from ipfx.stimulus import StimulusOntology
from ipfx.aibs_data_set import AibsDataSet
from ipfx.aibs_data_set import EphysDataSet
import ipfx.sweep_props as sp
from .helpers_for_tests import compare_dicts
import allensdk.core.json_utilities as ju


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_validate_required_sweep_info(NWB_file):

    sweep_info = [{"sweep_number": 0}]
    dataset = AibsDataSet(sweep_info, nwb_file=NWB_file, api_sweeps=False)

    assert list(dataset.sweep_table).sort() == dataset.COLUMN_NAMES.sort()


def test_modify_sweep_info_keys():
    d = [{"sweep_number": 123,
          "stimulus_units": "abcd",
          "stimulus_absolute_amplitude": 456,
          "stimulus_description": "efgh[4711]",
          "stimulus_name": "hijkl",
          }]

    result = sp.modify_sweep_info_keys(d)

    expected = [{EphysDataSet.SWEEP_NUMBER: 123,
                 EphysDataSet.STIMULUS_UNITS: "abcd",
                 EphysDataSet.STIMULUS_AMPLITUDE: 456,
                 EphysDataSet.STIMULUS_CODE: "efgh",
                 EphysDataSet.STIMULUS_NAME: "hijkl",
                 }]

    assert len(expected) == len(result)
    compare_dicts(expected[0], result[0])


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_get_clamp_mode(NWB_file):

    default_ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))

    dataset = AibsDataSet(nwb_file=NWB_file, ontology=default_ontology)

    assert dataset.get_clamp_mode(0) == dataset.VOLTAGE_CLAMP


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_get_stimulus_units(NWB_file):

    default_ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    dataset = AibsDataSet(nwb_file=NWB_file, ontology=default_ontology)
    assert dataset.get_stimulus_units(0) == "Volts"


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_get_stimulus_code(NWB_file):

    default_ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    dataset = AibsDataSet(nwb_file=NWB_file, ontology=default_ontology)
    assert dataset.get_stimulus_code(0) == "EXTPSMOKET180424"


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_get_stimulus_code_ext(NWB_file):
    default_ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    dataset = AibsDataSet(nwb_file=NWB_file, ontology=default_ontology)

    assert dataset.get_stimulus_code_ext("EXTPSMOKET180424",0) == "EXTPSMOKET180424[0]"
