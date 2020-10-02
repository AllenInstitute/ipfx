import pandas as pd
from typing import Tuple
import os

PARENT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PARENT_DIR, "data_release")
ARCHIVE_INFO = pd.DataFrame(
    {
        "dataset": ["human", "mouse"],
        "size (GB)": [12,114],
        "archive_url": [
            "https://dandiarchive.org/dandiset/000023",
            "https://dandiarchive.org/dandiset/000020",
        ],
        "file_manifest_path":[
            os.path.join(DATA_DIR, "2020-06-26_human_file_manifest.csv"),
            os.path.join(DATA_DIR, "2020-06-26_mouse_file_manifest.csv"),
        ],
        "experiment_metadata_path": [
            os.path.join(DATA_DIR, "20200625_patchseq_metadata_human.csv"),
            os.path.join(DATA_DIR, "20200625_patchseq_metadata_mouse.csv")
        ],
    }
).set_index("dataset")


def get_archive_info(
        dataset: str, 
        archive_info:pd.DataFrame = ARCHIVE_INFO
)-> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Provide information about released archive

    Parameters
    ----------
    dataset : name of the dataset to query. Currently supported options are:
        - human
        - mouse
    archive_info : dataframe of metadata and manifest files for each supported 
        dataset. Dataset name is the index. 

    Returns
    -------
    Information about the archive
    """

    if dataset in archive_info.index.values:
        file_manifest_path = archive_info.at[dataset, "file_manifest_path"]
        metadata_path = archive_info.at[dataset, "experiment_metadata_path"]
        archive_url = archive_info.at[dataset, "archive_url"]
    else:
        raise ValueError(
            f"No archive for the dataset '{dataset}'. Choose from the known "
            f"datasets: {archive_info.index.values}"
        )

    file_manifest = pd.read_csv(file_manifest_path)
    experiment_metadata = pd.read_csv(metadata_path)
    return archive_url, file_manifest, experiment_metadata
