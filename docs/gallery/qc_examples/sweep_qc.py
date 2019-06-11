"""
Sweep QC Features
=================

Estimate sweep QC features
"""
from __future__ import print_function

import os
import pandas as pd
from ipfx.aibs_data_set import AibsDataSet
import ipfx.qc_features as qcf
from allensdk.api.queries.cell_types_api import CellTypesApi


specimen_id = 595570553
nwb_file = '%d.nwb' % specimen_id

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

data_set = AibsDataSet(sweep_info=sweep_info, nwb_file=nwb_file)

# run sweep QC
sweep_features = qcf.sweep_qc_features(data_set)

print(pd.DataFrame(sweep_features).head())
