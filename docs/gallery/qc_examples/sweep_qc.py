"""
Sweep QC Features
=================

Estimate sweep QC features
"""

import os
import pandas as pd
from allensdk.ipfx.aibs_data_set import AibsDataSet
import allensdk.ipfx.qc_features as qcf
from allensdk.api.queries.cell_types_api import CellTypesApi


specimen_id = 595570553
nwb_file = '%d.nwb' % specimen_id

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweeps = ct.get_ephys_sweeps(specimen_id)

data_set = AibsDataSet(sweeps, nwb_file)

# run sweep QC
sweep_features = qcf.sweep_qc_features(data_set)

print(pd.DataFrame(sweep_features).head())
