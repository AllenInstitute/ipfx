import logging
import os
import json

import argschema as ags
import numpy as np

import ipfx.lims_queries as lq


class CollectFeatureVectorParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(
        description="Destination directory for output files",
        default="."
    )
    specimen_id_file = ags.fields.InputFile(
        description=("Input file of specimen IDs (one per line)"),
    )
    nwb_path_file = ags.fields.InputFile(
        description=("JSON file with paths to each specimen's NWB file - "
            "if not supplied, LIMS will be queried for them"),
        default=None,
        allow_none=True,
    )
    sweep_qc_record_file = ags.fields.InputFile(
        description=("File with sweep QC status and tags - "
            "if not supplied, LIMS will be queried for them"),
        default=None,
        allow_none=True,
    )
    manual_fail_sweep_file = ags.fields.InputFile(
        description=("File with manual sweep failure information"),
    )
    sweep_qc_option = ags.fields.String(
        description=("Sweep-level QC option - "
            "'none': use all sweeps; "
            "'lims-passed-only': check passed status with LIMS and "
            "only used passed sweeps "
            "'lims-passed-except-delta-vm': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm"
            "'lims-passed-except-delta-vm-and-rms': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm,"
            "but also re-calculate RMS values with current code"
            ),
        default='none'
    )


def main(args):
    ids = np.genfromtxt(args["specimen_id_file"], dtype=int).tolist()

    nwb_path_file = args["nwb_path_file"]
    if nwb_path_file is None:
        file_list = lq.get_nwb_file_paths_for_specimen_ids(ids)
    else:
        with open(nwb_path_file, "r") as f:
            file_list = json.load(f)

    print(file_list)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)
    main(module.args)
