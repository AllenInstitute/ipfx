from typing import Optional, Dict, Any
import h5py
from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.hbg_nwb_data import HBGNWBData
from ipfx.dataset.mies_nwb_data import MIESNWBData


def is_file_mies(path: str) -> bool:
    with h5py.File(path, "r") as fil:
        if "generated_by" in fil["general"].keys():
            generated_by = dict(fil["general"]["generated_by"][:])
            return generated_by.get("Package", "None") == "MIES"

    return False


def get_nwb_version(nwb_file: str) -> Dict[str, Any]:
    """
    Find version of the nwb file

    Parameters
    ----------
    nwb_file

    Returns
    -------

    dict in the format:

    {
        `major`: str
        `full` str.
    }

    """

    with h5py.File(nwb_file, 'r') as f:
        if "nwb_version" in f:         # In version 0 and 1 this is a dataset
            nwb_version = get_scalar_value(f["nwb_version"][()])
            nwb_version_str = to_str(nwb_version)
            if nwb_version is not None and re.match("^NWB-", nwb_version_str):
                return {"major": int(nwb_version_str[4]), "full": nwb_version_str}

        elif "nwb_version" in f.attrs:   # but in version 2 this is an attribute
            nwb_version = f.attrs["nwb_version"]
            if nwb_version is not None and (re.match("^2", nwb_version) or re.match("^NWB-2", nwb_version)):
                return {"major": 2, "full": nwb_version}

    return {"major": None, "full": None}


def create_ephys_data_set(nwb_file: str,
                          sweep_info: Optional[Dict[str,Any]],
                          ontology_file: Optional[str]) -> EphysDataSet:

    """
    Create an ephys data set with the appropriate nwbdata reader class

    Parameters
    ----------
    nwb_file
    sweep_info
    ontology_file

    Returns
    -------

    EphysDataSet

    """
    nwb_version = get_nwb_version(nwb_file)
    is_mies = is_file_mies(nwb_file)

    if not ontology_file:
        ontology_file = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE

    if nwb_version["major"] == 2:
        if is_mies:
            nwb_data = MIESNWBData(nwb_file, ontology_file)
        else:
            nwb_data = HBGNWBData(nwb_file, ontology_file)

    else:
        raise ValueError("Unsupported or unknown NWB major" +
                         "version {} ({})".format(nwb_version["major"], nwb_version["full"]))


    return EphysDataSet(sweep_info=sweep_info,
                        data=nwb_data,
                        )
