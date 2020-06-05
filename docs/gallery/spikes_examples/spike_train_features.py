"""
Spike Train Features
====================

Detect spike train features
"""

import os
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)

# Download and access the experimental data
nwb_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)

# Create data set from the nwb file and choose a sweeep
dataset = create_ephys_data_set(nwb_file=nwb_file)
sweep = dataset.sweep(sweep_number=39)

# Instantiate feature extractor for spikes
start, end = 1.02, 2.02
sfx = SpikeFeatureExtractor(start=start, end=end)

# Run feature extractor returning a table of spikes and their features
spikes_df = sfx.process(t=sweep.t, v=sweep.v, i=sweep.i)

# Instantiate Spike Train feature extractor
stfx = SpikeTrainFeatureExtractor(start=start, end=end)

# Run to produce features of a spike train
spike_train_results = stfx.process(
    t=sweep.t,
    v=sweep.v,
    i=sweep.i,
    spikes_df=spikes_df
)

print(spike_train_results)
