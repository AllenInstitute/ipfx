from ipfx.dataset.create import is_file_mies, create_ephys_data_set

TEST_MIES_FILE_PATH = 'tests/data/nwb/Ctgf-T2A-dgCre;Ai14-495723.05.02.01.nwb'
TEST_NON_MIES_FILE_PATH = 'tests/data/2018_03_20_0005.nwb'

def test_is_file_mies():
    assert is_file_mies(TEST_MIES_FILE_PATH) == True

def test_is_file_mies_with_non_mies_file():
    """
    Tests the case where the file is not detected as a MIES file.
    The file is read using the ipfx.dataset.hbg_nwb_data.HBGNWBData class.
    """
    assert is_file_mies(TEST_NON_MIES_FILE_PATH) == False
    nwb_data = create_ephys_data_set(TEST_NON_MIES_FILE_PATH)
    assert nwb_data is not None
    assert len(nwb_data.ontology.stimuli) > 0