"""
Sweep QC Features
=================

Estimate sweep QC features
"""
from __future__ import print_function

import os
import pandas as pd
from ipfx.data_set_utils import create_data_set
from ipfx.qc_feature_extractor import sweep_qc_features
from allensdk.api.queries.cell_types_api import CellTypesApi

# Download and access the experimental data
ct = CellTypesApi()
nwb_file = os.path.join(
    os.path.dirname(os.getcwd()), 
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)
specimen_id = 595570553
sweep_info = ct.get_ephys_sweeps(specimen_id)

data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)

# run sweep QC
sweep_features = sweep_qc_features(data_set)

print(pd.DataFrame(sweep_features).head())
