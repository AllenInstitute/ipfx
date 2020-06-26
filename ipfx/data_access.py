import pandas as pd
from typing import Tuple
import os

ARCHIVE_INFO = pd.DataFrame(

    {
        "organism": ["human", "mouse"],
        "size (GB)": [12,114],
        "archive_url": [
            "https://dandiarchive.org/dandiset/000023",
            "https://dandiarchive.org/dandiset/000020",
        ],
        "file_manifest_path":[
            os.path.join(os.path.dirname(__file__), "../data_release/2020-06-15_human_file_manifest.csv"),
            os.path.join(os.path.dirname(__file__), "../data_release/2020-06-15_mouse_file_manifest.csv"),
        ],
        "experiment_metadata_path": [
            os.path.join(os.path.dirname(__file__), "../data_release/20200625_patchseq_metadata_human_need_t-types.csv"),
            os.path.join(os.path.dirname(__file__), "../data_release/20200611_aibs_patchseq_metadata_mouse.csv")
        ],
    }
)


def get_archive_info(organism: str)-> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Provide information about released archive

    Parameters
    ----------
    organism : name of the organism

    Returns
    -------
    Information about the archive
    """
    archives_df = ARCHIVE_INFO.set_index("organism")

    if organism in archives_df.index.values:
        file_manifest = pd.read_csv(archives_df.loc[organism]["file_manifest_path"])
        experiment_metadata = pd.read_csv(archives_df.loc[organism]["experiment_metadata_path"])
        archive_url = archives_df.loc[organism]["archive_url"]
    else:
        raise ValueError(f"No archive for the organism '{organism}'. "
                         f"Choose from the known organisms: {archives_df.index.values}")
    return archive_url, file_manifest, experiment_metadata

