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
shsq_sweep_set = data_set.sweep_set(shsq_table.sweep_number)

# Estimate the dv cutoff and threshold fraction 
# (we know stimulus starts at 0.27s)
dv_cutoff, thresh_frac = estimate_adjusted_detection_parameters(
    shsq_sweep_set.v, shsq_sweep_set.t, 0.27, 0.271
)
# Build the extractors 
start = 0.27
spx = SpikeFeatureExtractor(
    start=start, dv_cutoff=dv_cutoff, thresh_frac=thresh_frac
)
sptrx = SpikeTrainFeatureExtractor(start=start, end=None)

# Run the analysis
shsq_analysis = ShortSquareAnalysis(spx, sptrx)
results = shsq_analysis.analyze(shsq_sweep_set)

# Plot the sweeps at the lowest amplitude that evoked the most spikes
for i, swp in enumerate(shsq_sweep_set.sweeps):
    if i in results["common_amp_sweeps"].index:
        plt.plot(swp.t, swp.v, linewidth=0.5, color="steelblue")

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(0.265, 0.3)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")
plt.title("Lowest amplitude spiking sweeps")

plt.show()
