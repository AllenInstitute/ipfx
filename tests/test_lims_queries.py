import pytest
import ipfx.lims_queries as lq

@pytest.mark.requires_lims
def test_get_specimen_info_from_lims_by_id():

    specimen_id = 500844783
    result = lq.get_specimen_info_from_lims_by_id(specimen_id)

    assert result == (u'Vip-IRES-Cre;Ai14(IVSCC)-226110.03.01.01', 500844779, 500844783)

@pytest.mark.requires_lims
def test_get_nwb_path_from_lims():

    ephys_roi_result = 500844779
    result = lq.get_nwb_path_from_lims(ephys_roi_result)

    assert result == "/allen/programs/celltypes/production/mousecelltypes/prod589/Ephys_Roi_Result_500844779/500844779.nwb"

@pytest.mark.requires_lims
def test_get_igorh5_path_from_lims():

    ephys_roi_result = 500844779

    result = lq.get_igorh5_path_from_lims(ephys_roi_result)
    assert result == "/allen/programs/celltypes/production/mousecelltypes/prod589/Ephys_Roi_Result_500844779/Vip-IRES-Cre;Ai14(IVSCC)-226110.03.01.h5"

