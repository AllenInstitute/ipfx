"""
Ramp Analysis
====================

Detect ramp features
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.epochs import get_stim_epoch

from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)
from ipfx.stimulus_protocol_analysis import RampAnalysis

# Download and access the experimental data
ct = CellTypesApi()
nwb_file = os.path.join(
    os.path.dirname(os.getcwd()), 
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)
raise ValueError(nwb_file)
specimen_id = 595570553
sweep_info = ct.get_ephys_sweeps(specimen_id)

# Build the data set and find the ramp sweeps
data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)
ramp_table = data_set.filtered_sweep_table(
    stimuli=data_set.ontology.ramp_names
)
ramp_sweeps = data_set.sweep_set(ramp_table.sweep_number)

# find the start and end time of the stimulus 
# (treating the first sweep as representative)
stim_start_index, stim_end_index = get_stim_epoch(ramp_sweeps.i[0])
stim_start_time = ramp_sweeps.t[0][stim_start_index]
stim_end_time = ramp_sweeps.t[0][stim_end_index]

spx = SpikeFeatureExtractor(start=stim_start_time, end=None)
sptrx = SpikeTrainFeatureExtractor(start=stim_start_time, end=None)

# Run the analysis
ramp_analysis = RampAnalysis(spx, sptrx)
results = ramp_analysis.analyze(ramp_sweeps)

# Plot the sweeps and the latency to the first spike of each
sns.set_style("white")
for swp in ramp_sweeps.sweeps:
    plt.plot(swp.t, swp.v, linewidth=0.5)
sns.rugplot(results["spiking_sweeps"]["latency"].values + stim_start_time)

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(stim_start_time, stim_end_time)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")

plt.show()
