"""
Spike Train Features
====================

Detect spike train features
"""
from __future__ import print_function

import os
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)

# Download and access the experimental data
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
sweep_info = ct.get_ephys_sweeps(specimen_id)

if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)

# Get the data for the sweep into a format we can use
dataset = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)
sweep_number = 39
sweep = dataset.sweep(sweep_number)

# Extract information about the spikes
start, end = 1.02, 2.02
ext = SpikeFeatureExtractor(start=start, end=end)
spikes_df = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)

st_ext = SpikeTrainFeatureExtractor(start=start, end=end)
st_results = st_ext.process(
    t=sweep.t, 
    v=sweep.v, 
    i=sweep.i, 
    spikes_df=spikes_df
)

print(st_results)
