import h5py

from ipfx.hbg_dataset import HBGDataSet
from ipfx.aibs_data_set import AibsDataSet
from ipfx.mies_data_set import MiesDataSet
from ipfx.nwb_reader import get_nwb_version


def is_file_mies(path: str) -> bool:
    with h5py.File(path, "r") as fil:
        if "generated_by" in fil["general"].keys():
            generated_by = dict(fil["general"]["generated_by"][:])
            return generated_by.get("Package", "None") == "MIES"

    return False


def create_data_set(sweep_info=None, nwb_file=None, ontology=None, api_sweeps=True, h5_file=None,validate_stim=True):
    """Create an appropriate EphysDataSet derived class for the given nwb_file

    Parameters
    ----------
    nwb_file: str file name

    Returns
    -------
    EphysDataSet derived object
    """

    if nwb_file is None:
        raise ValueError("Can not decide which EphysDataSet class to create without nwb_file")

    nwb_version = get_nwb_version(nwb_file)
    is_mies = is_file_mies(nwb_file)

    if nwb_version["major"] == 2 and is_mies:
        return MiesDataSet(
            sweep_info=sweep_info,
            nwb_file=nwb_file,
            ontology=ontology,
            api_sweeps=api_sweeps,
            validate_stim=validate_stim
        )

    elif nwb_version["major"] == 2:
        return HBGDataSet(sweep_info=sweep_info,
                          nwb_file=nwb_file,
                          ontology=ontology,
                          api_sweeps=api_sweeps,
                          validate_stim=validate_stim)

    elif nwb_version["major"] == 1 or nwb_version["major"] == 0:
        return AibsDataSet(sweep_info=sweep_info,
                           nwb_file=nwb_file,
                           ontology=ontology,
                           api_sweeps=api_sweeps,
                           h5_file=h5_file,
                           validate_stim=validate_stim)
    else:
        raise ValueError("Unsupported or unknown NWB major" +
                         "version {} ({})".format(nwb_version["major"], nwb_version["full"]))
