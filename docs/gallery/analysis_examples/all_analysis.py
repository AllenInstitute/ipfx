"""
All Analysis
============

Run all analyses on NWB file
"""
from __future__ import print_function

import os
import warnings

import numpy as np

from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.data_set_features import extract_data_set_features

warnings.filterwarnings(
    "ignore", category=np.VisibleDeprecationWarning,
    message=(
        "NWB1 support is deprecated for ipfx 1.0.0, but we will release an "
        "NWB2 version of the data used in this example."
    )
)

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)

# Download extracted sweeps, excluding any without a proper stimulus presented
sweep_info = ct.get_ephys_sweeps(specimen_id)
sweep_info = [
    sweep for sweep in sweep_info 
    if sweep["stimulus_name"] != "Test"
]

data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)

cell_features, sweep_features, cell_record, sweep_records = \
    extract_data_set_features(data_set, subthresh_min_amp=-100.0)

print(cell_record)
