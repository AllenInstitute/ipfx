"""
Ramp Analysis
====================

Detect ramp features
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.aibs_data_set import AibsDataSet
from ipfx.ephys_extractor import SpikeExtractor, SpikeTrainFeatureExtractor
from ipfx.stimulus_protocol_analysis import RampAnalysis

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

# Build the data set and find the ramp sweeps
data_set = AibsDataSet(sweep_info=sweep_info, nwb_file=nwb_file)
ramp_table = data_set.filtered_sweep_table(stimuli=dataset.ontology.ramp_names)
ramp_sweep_set = data_set.sweep_set(ramp_table.sweep_number)

# Build the extractors (we know stimulus starts at 1.02 s)
start = 1.02
spx = SpikeExtractor(start=start, end=None)
sptrx = SpikeTrainFeatureExtractor(start=start, end=None)

# Run the analysis
ramp_analysis = RampAnalysis(spx, sptrx)
results = ramp_analysis.analyze(ramp_sweep_set)

# Plot the sweeps and the latency to the first spike of each
sns.set_style("white")
for swp in ramp_sweep_set.sweeps:
    plt.plot(swp.t, swp.v, linewidth=0.5)
sns.rugplot(results["spiking_sweeps"]["latency"].values + start)

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(0, 11)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")
