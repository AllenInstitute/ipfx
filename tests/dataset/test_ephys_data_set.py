from typing import Dict, Any

import pytest
import numpy as np
import pandas as pd

from ipfx.dataset.ephys_data_set import EphysDataSet, _nan_trailing_zeros
from ipfx.dataset.ephys_data_interface import EphysDataInterface 

@pytest.mark.parametrize("base_arr", [
    np.arange(20, dtype=int), np.arange(10, dtype=float)])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("num_trailing", np.arange(5))
def test_nan_trailing_zeros(base_arr, inplace, num_trailing):

    expected = np.concatenate([base_arr, np.zeros((num_trailing)) * np.nan])
    inputs = np.concatenate([base_arr, np.zeros((num_trailing))])

    obtained = _nan_trailing_zeros(inputs, inplace=inplace)
    assert np.allclose(np.isnan(expected), np.isnan(obtained))
    if inplace:
        np.allclose(np.isnan(inputs), np.isnan(obtained))


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


class EphysDataFixture(EphysDataInterface):
    """
    """

    sweep_meta = {
        1: {
            "sweep_number": 1,
            "stimulus_units": "amperes",
            "bridge_balance_mohm": "1.0",
            "leak_pa": "0.0",
            "stimulus_scale_factor": "1.0",
            "stimulus_code": "astim",
            "stimulus_code_ext": "astim[1]",
            "stimulus_name": "a stimulus",
            "clamp_mode": "CurrentClamp"
        },
        2: {
            "sweep_number": 2,
            "stimulus_units": "amperes",
            "bridge_balance_mohm": "1.0",
            "leak_pa": "0.0",
            "stimulus_scale_factor": "1.0",
            "stimulus_code": "astim",
            "stimulus_code_ext": "astim[2]",
            "stimulus_name": "a stimulus",
            "clamp_mode": "CurrentClamp"
        }
    }

    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        return self.sweep_meta[sweep_number]


def test_sweep_table():
    dataset = EphysDataSet(EphysDataFixture())

    expected = pd.DataFrame(EphysDataFixture.sweep_meta.values())
    pd.testing.assert_frame_equal(
        expected, dataset.sweep_table, check_like=True
    )
