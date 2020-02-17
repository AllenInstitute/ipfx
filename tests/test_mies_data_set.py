import pytest

from ipfx.mies_data_set import MiesDataSet


class MiesDs(MiesDataSet):

    @property
    def nwb_data(self):
        return self._nwb_data

    @nwb_data.setter
    def nwb_data(self, val):
        self._nwb_data = val

    def __init__(self):
        """ This class does work on __init__, so this hack is used to inject 
        nwb_data & other dependencies.
        """
        # TODO: when refactored, this class ought ot have its dependencies 
        # injected


def test_get_stimulus_code_ext():

    class NwbData:
        def get_stim_code(self, sweep_num):
            return "fizz"
    
    class Notebook:
        def get_value(self, key, sweep_num, default):
            return {
                ("Set Sweep Count", 12): "100"
            }.get((key, sweep_num), default)

    ds = MiesDs()
    ds.nwb_data = NwbData()
    ds.notebook = Notebook()

    assert ds.get_stimulus_code_ext(12) == "fizz[100]"


def test_extract_sweep_record():

    class NwbData:

        def get_stim_code(self, sweep_num):
            return "fizz"

        def get_stimulus_unit(self, sweep_num):
            return "amperes"

        def get_sweep_attrs(self, sweep_num):
            return {
                "bridge_balance": 100.0,
                "bias_current": 200.0,
            }

    class Notebook:

        def get_value(self, key, sweep_num, default):
            return {
                ("Scale Factor", 12): 2,
                ("Set Sweep Count", 12): "5"
            }.get((key, sweep_num), default)

    sweep = 12

    exp = {
        "sweep_number": sweep,
        "stimulus_units": "amperes",
        "bridge_balance_mohm": 100.0,
        "leak_pa": 200.0,
        "stimulus_scale_factor": 2,
        "stimulus_code": "fizz",
        "stimulus_code_ext": "fizz[5]"
    }

    ds = MiesDs()
    ds.nwb_data = NwbData()
    ds.notebook = Notebook()
    ds.ontology = None

    obt = ds.extract_sweep_record(sweep)

    misses = []
    for key in exp.keys():
        obt_val = obt.pop(key)
        if not exp[key] == obt_val:
            misses.append([exp[key], obt_val])

    assert len(obt) == 0 and len(misses) == 0, f"{misses}\n{obt}"