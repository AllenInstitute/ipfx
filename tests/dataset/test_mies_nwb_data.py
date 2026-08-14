import pytest
import pynwb
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.mies_nwb_data import MIESNWBData
from tests.dataset.test_ephys_nwb_data import nwbfile_to_test
from ipfx.dataset.labnotebook import LabNotebookReader, LabNotebookReaderIgorNwb

# uint64 conversion warning pynwb/hdmf raises when writing the small int
# timestamps/indices used throughout these synthetic test fixtures.
IGNORE_UINT64_WARNING = (
    "ignore:.*Value with data type int64 is being converted to data type "
    "uint64.*"
)


def nwbfile_with_epochs_to_test():
    """
    Same single-sweep (sweep_number=4) nwbfile as nwbfile_to_test(), plus a
    MIES-style epochs (nwbfile.epochs) table with two rows referencing that
    sweep's stimulus series -- for exercising MIESNWBData.get_nwb_epochs.
    """

    nwbfile = nwbfile_to_test()
    stimulus_series = nwbfile.get_stimulus("stimulus")

    dt = 1.0 / stimulus_series.rate
    starting_time = stimulus_series.starting_time

    # pynwb computes each row's idx_start/count as int((time - starting_time)
    # * rate) -- i.e. plain truncation, not rounding.
    nwbfile.add_epoch_column(
        name="treelevel", description="MIES epoch tree level"
    )
    nwbfile.add_epoch(
        start_time=starting_time,
        stop_time=starting_time + 2.1 * dt,
        tags=[
            "Type=Epoch", "Epoch=0", "EpochType=Square pulse",
            "ShortName=E0",
        ],
        timeseries=[stimulus_series],
        treelevel=1,
    )
    nwbfile.add_epoch(
        start_time=starting_time + 2.1 * dt,
        stop_time=starting_time + 4.1 * dt,
        tags=[
            "Type=Epoch", "Epoch=1", "EpochType=Square pulse",
            "ShortName=E1",
        ],
        timeseries=[stimulus_series],
        treelevel=1,
    )

    return nwbfile


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


@pytest.mark.filterwarnings(IGNORE_UINT64_WARNING)
def test_create_mies(mies_nwb_data):
    assert isinstance(mies_nwb_data, MIESNWBData)


@pytest.mark.filterwarnings(IGNORE_UINT64_WARNING)
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


@pytest.mark.filterwarnings(IGNORE_UINT64_WARNING)
def test_get_nwb_epochs_no_epochs_table(mies_nwb_data):
    """A file with no nwbfile.epochs table at all (the common mies_nwb_data
    fixture doesn't add one) must return an empty list, not raise."""

    assert mies_nwb_data.get_nwb_epochs(sweep_number=4) == []


@pytest.fixture
def mies_nwb_data_with_epochs(tmp_nwb_path):

    nwbfile = nwbfile_with_epochs_to_test()

    with pynwb.NWBHDF5IO(path=tmp_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    ontology = StimulusOntology(
        [[('name', 'expected name'), ('code', 'STIMULUS_CODE')],
         [('name', 'test name'), ('code', 'extpexpend')]
         ])

    class Notebook(LabNotebookReader):

        def get_value(self, key, sweep_num, default):
            return {
                ("Scale Factor", 4): 200.0,
                ("Set Sweep Count", 4): "1"
            }.get((key, sweep_num), default)

    return MIESNWBData(nwb_file=tmp_nwb_path,
                       notebook=Notebook(),
                       ontology=ontology)


@pytest.mark.filterwarnings(IGNORE_UINT64_WARNING)
def test_get_nwb_epochs_returns_matching_rows(mies_nwb_data_with_epochs):

    obtained = mies_nwb_data_with_epochs.get_nwb_epochs(sweep_number=4)

    assert len(obtained) == 2

    short_names = [rec["tags"]["ShortName"] for rec in obtained]
    assert short_names == ["E0", "E1"]

    e0, e1 = obtained

    assert e0["start_idx"] == 0
    assert e0["end_idx"] == 2
    assert e0["treelevel"] == 1
    assert e0["tags"]["Type"] == "Epoch"
    assert e0["tags"]["Epoch"] == "0"
    assert e0["tags"]["EpochType"] == "Square pulse"
    assert e0["name"] == "nwb:E0"

    assert e1["start_idx"] == 2
    assert e1["end_idx"] == 4
    assert e1["tags"]["Epoch"] == "1"
    assert e1["name"] == "nwb:E1"


@pytest.mark.filterwarnings(IGNORE_UINT64_WARNING)
def test_get_nwb_epochs_no_match_for_other_sweep(mies_nwb_data_with_epochs):
    """The epochs table has rows, but none reference a sweep number that
    isn't actually present in the file -- must return an empty list, not
    raise, since get_series would itself fail differently for a genuinely
    unknown sweep."""

    with pytest.raises(ValueError):
        # sweep 4 is the only sweep in this fixture file
        mies_nwb_data_with_epochs.get_nwb_epochs(sweep_number=999)
