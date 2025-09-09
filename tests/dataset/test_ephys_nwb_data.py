import pytest
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.ephys_nwb_data import EphysNWBData
import ipfx.json_utilities as ju
import datetime
import pynwb
from pynwb.icephys import CurrentClampStimulusSeries, CurrentClampSeries
import numpy as np
from ipfx.utilities import inject_sweep_table

from dictdiffer import diff

@pytest.fixture
def tmp_nwb_path(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test_nwb_data").join("test_ephys_data.nwb")
    return str(nwb)


def nwbfile_to_test():
    """
    Create a simple nwbfile for testing

    Returns
    -------
    nwbfile
    """

    nwbfile = pynwb.NWBFile(
        session_description="test nwb data",
        identifier='test session',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    device = nwbfile.create_device(name='electrode_0')

    electrode = nwbfile.create_icephys_electrode(
        name="electrode 0",
        description='fancy electrode',
        device=device)

    stimulus_data = [0, 3, 3, 3, 0]

    stimulus_meta_data = {
        "name": "stimulus",
        "sweep_number": 4,
        "unit": "amperes",
        "gain": 32.0,
         "resolution": 1.0,
         "conversion": 1.0E-3,
         "starting_time": 1.5,
         "rate": 7000.0,
         "stimulus_description": "STIMULUS_CODE"
    }

    stimulus_series = CurrentClampStimulusSeries(
        data=stimulus_data,
        electrode=electrode,
        **stimulus_meta_data
    )

    inject_sweep_table(nwbfile)
    nwbfile.add_stimulus(stimulus_series, use_sweep_table=True)

    response_data = [1, 2, 3, 4, 5]
    response_meta_data = {
        "name":"acquisition",
         "sweep_number": 4,
         "unit": "volts",
         "gain": 32.0,
         "resolution": 1.0,
         "conversion": 1.0E-3,
         "bridge_balance": 500.0,
         "bias_current":100.0,
         "starting_time": 1.5,
         "rate": 7000.0,
         "stimulus_description": "STIMULUS_CODE"
    }

    acquisition_series = CurrentClampSeries(data=response_data,
                                             electrode=electrode,
                                             **response_meta_data
                                             )

    nwbfile.add_acquisition(acquisition_series, use_sweep_table=True)

    return nwbfile


@pytest.fixture
def nwb_data(tmp_nwb_path):

    nwbfile = nwbfile_to_test()
    with pynwb.NWBHDF5IO(path=tmp_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    ontology =  StimulusOntology(
        [[('name', 'expected name'), ('code', 'STIMULUS_CODE')],
         [('name', 'test name'), ('code', 'extpexpend')]
         ])

    return EphysNWBData(nwb_file=tmp_nwb_path, ontology=ontology)


def test_get_stimulus_unit(nwb_data):
    assert nwb_data.get_stimulus_unit(sweep_number=4) == "Amps"


def test_get_stimulus_code(nwb_data):
    assert nwb_data.get_stimulus_code(sweep_number=4) == "STIMULUS_CODE"


def test_get_sweep_data(nwb_data):

    expected = {
        'stimulus': np.array([0, 3, 3, 3, 0], dtype=float) * 1.0e9,
        'response': np.array([1, 2, 3, 4, 5], dtype=float),
        'sampling_rate': 7000.0,
        'stimulus_unit': 'Amps'
    }

    obtained = nwb_data.get_sweep_data(sweep_number=4)

    assert list(diff(expected, obtained, tolerance=0.001)) == []


def test_get_sweep_attrs(nwb_data):

    expected = {
        'stimulus_description': 'STIMULUS_CODE',
        'sweep_number': 4,
        'bias_current': 100.0,
        'bridge_balance': 500.0,
        'gain': 32.0,
        'clamp_mode': 'CurrentClamp',
    }

    obtained = nwb_data.get_sweep_attrs(sweep_number=4)
    print(obtained)
    print(expected)
    assert expected == obtained

def test_get_clamp_mode(nwb_data):

    attrs = nwb_data.get_sweep_attrs(4);

    assert attrs['clamp_mode'] == "CurrentClamp"

def test_get_full_recording_date(nwb_data):
    assert nwb_data.get_full_recording_date() == nwb_data.nwb.session_start_time

