"""
Short Square Analysis
=====================

Detect short square features
"""

import os
import matplotlib.pyplot as plt
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)
from ipfx.stimulus_protocol_analysis import ShortSquareAnalysis
from ipfx.spike_features import estimate_adjusted_detection_parameters
from ipfx.epochs import get_stim_epoch

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

# build a data set and find the short squares
data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)
shsq_table = data_set.filtered_sweep_table(
    stimuli=data_set.ontology.short_square_names
)
sweeps = data_set.sweep_set(shsq_table.sweep_number)

# find the start and end time of the stimulus 
# (treating the first sweep as representative)
stim_start_index, stim_end_index = get_stim_epoch(sweeps.i[0])
stim_start_time = sweeps.t[0][stim_start_index]
stim_end_time = sweeps.t[0][stim_end_index]

# Estimate the dv cutoff and threshold fraction 
dv_cutoff, thresh_frac = estimate_adjusted_detection_parameters(
    sweeps.v, 
    sweeps.t, 
    stim_start_time,
    stim_start_time + 0.001
)
# Build the extractors 

spx = SpikeFeatureExtractor(
    start=stim_start_time, dv_cutoff=dv_cutoff, thresh_frac=thresh_frac
)
sptrx = SpikeTrainFeatureExtractor(start=stim_start_time, end=None)

# Run the analysis
shsq_analysis = ShortSquareAnalysis(spx, sptrx)
results = shsq_analysis.analyze(sweeps)

# Plot the sweeps at the lowest amplitude that evoked the most spikes
for i, swp in enumerate(sweeps.sweeps):
    if i in results["common_amp_sweeps"].index:
        plt.plot(swp.t, swp.v, linewidth=0.5, color="steelblue")

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(stim_start_time - 0.05, stim_end_time + 0.05)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")
plt.title("Lowest amplitude spiking sweeps")


plt.show()
