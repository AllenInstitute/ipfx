from ipfx.dataset.create import is_file_mies

TEST_NWB_FILE_PATH = 'tests/data/nwb/Ctgf-T2A-dgCre;Ai14-495723.05.02.01.nwb'

def test_is_file_mies():
    assert is_file_mies(TEST_NWB_FILE_PATH) == True