"""
All Analysis
============

Run all analyses on NWB file
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.aibs_data_set import AibsDataSet
from ipfx.data_set_features import extract_data_set_features

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

data_set = AibsDataSet(sweep_info=sweep_info, nwb_file=nwb_file)# Download and access the experimental data

cell_features, sweep_features, cell_record, sweep_records = \
    extract_data_set_features(data_set, subthresh_min_amp=-100.0)

print(cell_record)
