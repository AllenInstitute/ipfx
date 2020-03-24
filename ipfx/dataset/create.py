from typing import Optional, Dict, Any
import re
from pathlib import Path

import h5py
import numpy as np

import allensdk.core.json_utilities as ju

from ipfx.dataset.ephys_data_set import EphysDataSet
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.hbg_nwb_data import HBGNWBData
from ipfx.dataset.mies_nwb_data import MIESNWBData
from ipfx import py2to3
from ipfx.dataset.labnotebook import LabNotebookReaderIgorNwb


def get_scalar_value(dataset_from_nwb):
    """
    Some values in NWB are stored as scalar whereas others as np.ndarrays with 
    dimension 1. Use this function to retrieve the scalar value itself.
    """

    if isinstance(dataset_from_nwb, np.ndarray):
        return dataset_from_nwb.item()

    return dataset_from_nwb


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
            nwb_version_str = py2to3.to_str(nwb_version)
            if nwb_version is not None and re.match("^NWB-", nwb_version_str):
                return {
                    "major": int(nwb_version_str[4]), 
                    "full": nwb_version_str
                }

        elif "nwb_version" in f.attrs:   # in version 2 this is an attribute
            nwb_version = f.attrs["nwb_version"]
            if nwb_version is not None and (
                    re.match("^2", nwb_version) or
                    re.match("^NWB-2", nwb_version)
            ):
                return {"major": 2, "full": nwb_version}

    return {"major": None, "full": None}


def create_ephys_data_set(
        nwb_file: str,
        sweep_info: Optional[Dict[str, Any]] = None,
        ontology: Optional[str] = None
) -> EphysDataSet:
    """
    Create an ephys data set with the appropriate nwbdata reader class

    Parameters
    ----------
    nwb_file
    sweep_info
    ontology

    Returns
    -------

    EphysDataSet

    """
    nwb_version = get_nwb_version(nwb_file)
    is_mies = is_file_mies(nwb_file)

    if not ontology:
        ontology = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
    if isinstance(ontology, (str, Path)):
        ontology = StimulusOntology(ju.read(ontology))

    if nwb_version["major"] == 2:
        if is_mies:
            labnotebook = LabNotebookReaderIgorNwb(nwb_file)
            nwb_data = MIESNWBData(nwb_file, labnotebook, ontology)
        else:
            nwb_data = HBGNWBData(nwb_file, ontology)

    else:
        raise ValueError(
            "Unsupported or unknown NWB major version {} ({})".format(
                nwb_version["major"], nwb_version["full"]
            )
        )

    return EphysDataSet(
        sweep_info=sweep_info,
        data=nwb_data,
    )
