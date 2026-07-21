import pytest
import pynwb
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.mies_nwb_data import MIESNWBData
from tests.dataset.test_ephys_nwb_data import nwbfile_to_test
from ipfx.dataset.labnotebook import LabNotebookReader, LabNotebookReaderIgorNwb


@pytest.fixture
def tmp_nwb_path(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test_nwb_data").join("test_mies_data.nwb")
    return str(nwb)


@pytest.fixture
def mies_nwb_data(tmp_nwb_path):

    nwbfile = nwbfile_to_test()
    print(tmp_nwb_path)

    with pynwb.NWBHDF5IO(path=tmp_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    ontology =  StimulusOntology(
        [[('name', 'expected name'), ('code', 'STIMULUS_CODE')],
         [('name', 'test name'), ('code', 'extpexpend')]
         ])


    class Notebook(LabNotebookReader):

        def get_value(self, key, sweep_num, default):
            return {
                ("Scale Factor", 4): 200.0,
                ("Set Sweep Count", 4): "1"
            }.get((key, sweep_num), default)

    fake_notebook = Notebook()

    return MIESNWBData(nwb_file=tmp_nwb_path,
                       notebook=fake_notebook,
                       ontology=ontology)


@pytest.mark.filterwarnings("ignore:.*Value with data type int64 is being converted to data type uint64.*")
def test_create_mies(mies_nwb_data):
    assert isinstance(mies_nwb_data, MIESNWBData)


@pytest.mark.filterwarnings("ignore:.*Value with data type int64 is being converted to data type uint64.*")
def test_get_sweep_metadata(mies_nwb_data):

    expected = {
        'sweep_number': 4,
        'stimulus_units': 'Amps',
        'bridge_balance_mohm': 500.0,
        'leak_pa': 100.0,
        'stimulus_scale_factor': 200.0,
        'stimulus_code': 'STIMULUS_CODE',
        'stimulus_code_ext': 'STIMULUS_CODE[1]',
        'clamp_mode': 'CurrentClamp',
        'stimulus_name': 'expected name',
    }

    obtained = mies_nwb_data.get_sweep_metadata(sweep_number=4)
    assert expected == obtained


