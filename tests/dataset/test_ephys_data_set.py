from typing import Dict, Any, Sequence
from datetime import datetime
import json

import pytest
import numpy as np
import pandas as pd

from ipfx.dataset.ephys_data_interface import EphysDataInterface
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.ephys_data_set import EphysDataSet


def test_voltage_current():

    stimulus = np.arange(5)
    response = np.arange(5, 10)

    obt_v, obt_i = EphysDataSet._voltage_current(
        stimulus, response, EphysDataSet.CURRENT_CLAMP
    )

    assert np.allclose(obt_v, response)
    assert np.allclose(obt_i, stimulus)


def test_voltage_current_unequal():
    with pytest.raises(ValueError):
        EphysDataSet._voltage_current(
            np.arange(2), np.arange(3), EphysDataSet.VOLTAGE_CLAMP
        )


with open(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE, "r") as _def_ont:
    _default_ont_data = json.load(_def_ont)
DEFAULT_ONT = StimulusOntology(_default_ont_data)


class EphysDataFixture(EphysDataInterface):
    """
    """

    REC_DATE = datetime.strptime(
        "2020-03-19 10:30:12 +1000", "%Y-%m-%d %H:%M:%S %z")

    SWEEPS: Dict[int, Dict[str, Dict[str, Any]]] = {
        1: {
            "meta": {
                "sweep_number": 1,
                "stimulus_units": "amperes",
                "bridge_balance_mohm": "1.0",
                "leak_pa": "0.0",
                "stimulus_scale_factor": "1.0",
                "stimulus_code": "PS_SupraThresh",
                "stimulus_name": "Long Square",
                "clamp_mode": "CurrentClamp"
            },
            "data": {
                "stimulus": np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0]),
                "response": np.arange(0, 5, 0.5),
                "sampling_rate": 1.5
            }
        },
        2: {
            "meta": {
                "sweep_number": 2,
                "stimulus_units": "amperes",
                "bridge_balance_mohm": "1.0",
                "leak_pa": "0.0",
                "stimulus_scale_factor": "1.0",
                "stimulus_code": "PS_SupraThresh",
                "stimulus_name": "Long Square",
                "clamp_mode": "CurrentClamp"
            },
            "data": {
                "stimulus": np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0]),
                "response": np.arange(10)[::-1],
                "sampling_rate": 1.5
            }
        },
        3: {
            "meta": {
                "sweep_number": 3,
                "stimulus_units": "volts",
                "bridge_balance_mohm": "1.0",
                "leak_pa": "0.0",
                "stimulus_scale_factor": "1.0",
                "stimulus_code": "shortsquaretemp",
                "stimulus_name": "Short Square",
                "clamp_mode": "VoltageClamp"
            },
            "data": {
                "stimulus": np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0]),
                "response": np.arange(10),
                "sampling_rate": 1.5
            }
        }
    }

    @property
    def sweep_numbers(self) -> Sequence[int]:
        return list(self.SWEEPS.keys())

    def get_sweep_data(self, sweep_number: int) -> Dict[str, Any]:
        return self.SWEEPS[sweep_number]["data"]

    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        return self.SWEEPS[sweep_number]["meta"]

    def get_sweep_attrs(self, sweep_number) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_stimulus_code(self, sweep_number: int) -> str:
        return self.SWEEPS[sweep_number]["meta"]["stimulus_code"]

    def get_full_recording_date(self) -> datetime:
        return self.REC_DATE

    def get_stimulus_unit(self, sweep_number: int) -> str:
        return self.SWEEPS[sweep_number]["meta"]["stimulus_units"]

    def get_clamp_mode(self, sweep_number):
        return self.SWEEPS[sweep_number]["meta"]["clamp_mode"]

@pytest.fixture
def dataset():
    return EphysDataSet(EphysDataFixture(DEFAULT_ONT))


def test_ontology(dataset):
    assert DEFAULT_ONT is dataset.ontology


def test_sweep_table(dataset):
    expected = pd.DataFrame([
        swp["meta"] for swp in EphysDataFixture.SWEEPS.values()
    ])
    pd.testing.assert_frame_equal(
        expected, dataset.sweep_table, check_like=True
    )


def test_filtered_sweep_table(dataset):
    expected = pd.DataFrame([
        swp["meta"]
        for num, swp in EphysDataFixture.SWEEPS.items()
        if num in {1, 2}
    ])
    pd.testing.assert_frame_equal(
        expected,
        dataset.filtered_sweep_table(clamp_mode=EphysDataSet.CURRENT_CLAMP),
        check_like=True
    )


def test_get_sweep_numbers(dataset):
    assert np.allclose(
        [1, 2],
        dataset.get_sweep_numbers(stimuli=["PS_SupraThresh"])
    )


def test_sweep(dataset):
    obtained = dataset.sweep(3)
    assert np.allclose([0, 1, 1, 0, 1, 1, 1, 1, 0, 0], obtained.v)
    assert np.allclose(np.arange(10), obtained.i)
    assert np.allclose(np.arange(10) / 1.5, obtained.t)


def test_sweep_set(dataset):
    sweepset = dataset.sweep_set([1, 3])
    assert np.allclose(
        [np.arange(0, 5, 0.5), np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0])],
        sweepset.v
    )


def test_get_recording_date(dataset):
    assert dataset.get_recording_date() == "2020-03-19 10:30:12"


def test_get_sweep_data(dataset):
    obtained = dataset.get_sweep_data(1)
    assert np.allclose(obtained["response"], np.arange(0, 5, 0.5))


def test_get_clamp_mode(dataset):
    assert "VoltageClamp" == dataset.get_clamp_mode(3)


def test_get_stimulus_code(dataset):
    assert "shortsquaretemp" == dataset.get_stimulus_code(3)


def test_get_stimulus_code_ext(dataset):
    assert "PS_SupraThresh[2]" == dataset.get_stimulus_code_ext(2)


def test_get_stimulus_units(dataset):
    assert "amperes" == dataset.get_stimulus_units(1)
