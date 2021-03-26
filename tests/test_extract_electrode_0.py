import pytest
import os
from ipfx.qc_feature_extractor import extract_electrode_0
from ipfx.dataset.create import create_ephys_data_set

def get_e0_tags(file_path, nwb_file, ont_file):
    nwb_filename = os.path.join(file_path, nwb_file)
    ont_filename = os.path.join(file_path, ont_file)

    data_set = create_ephys_data_set(nwb_file=nwb_filename, ontology=ont_filename)

    tags = []
    e0 = extract_electrode_0(data_set, tags)
    
    return e0, tags


def test_no_inbath_smoketest():
    # This experiment does not contain an EXTPSMOKET sweep where pipette was in bath
    file_path = "//allen/programs/celltypes/production/mousecelltypes/prod3053/Ephys_Roi_Result_1087546037/"
    nwb_file = "Vip-IRES-Cre;Ai14-567749.01.10.02.nwb"
    ont_file = "1087550462_stimulus_ontology.json"
    
    e0, tags = get_e0_tags(file_path, nwb_file, ont_file)
    assert e0 is None
    assert tags == ["Electrode 0 is not available"]

def test_inbath_smoketest():
    # This experiment does contain an EXTPSMOKET sweep where the pipette was in bath
    file_path = "//allen/programs/celltypes/production/mousecelltypes/prod3050/Ephys_Roi_Result_1085978679/"
    nwb_file = "Esr2-IRES2-Cre;Ai14-566833.04.10.02.nwb"
    ont_file = "1085980322_stimulus_ontology.json"

    e0, tags = get_e0_tags(file_path, nwb_file, ont_file)
    assert e0 == pytest.approx(56.88125)
    assert tags == []

def test_inbath_multiple_smoketests():
    # This experiment contains multiple EXTPSMOKET sweeps, but only 1 where pipette was in bath
    file_path = "//allen/programs/celltypes/production/mousecelltypes/prod3008/Ephys_Roi_Result_1077512171/"
    nwb_file = "Vipr2-IRES2-Cre;Slc32a1-IRES2-FlpO;Ai65-562071.12.09.04.nwb"
    ont_file = "1077513177_stimulus_ontology.json"

    e0, tags = get_e0_tags(file_path, nwb_file, ont_file)
    assert e0 == pytest.approx(41.40625)
    assert tags == []
