"""
Short Square Analysis
=====================

Detect short square features
"""

import os
import matplotlib.pyplot as plt
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.aibs_data_set import AibsDataSet
from ipfx.ephys_extractor import SpikeExtractor, SpikeTrainFeatureExtractor
from ipfx.stimulus_protocol_analysis import ShortSquareAnalysis
import ipfx.ephys_features as ft

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

# build a data set and find the short squares
data_set = AibsDataSet(sweep_info=sweep_info, nwb_file=nwb_file)
shsq_table = data_set.filtered_sweep_table(stimuli=data_set.ontology.short_square_names)
shsq_sweep_set = data_set.sweep_set(shsq_table.sweep_number)

# Estimate the dv cutoff and threshold fraction
dv_cutoff, thresh_frac = ft.estimate_adjusted_detection_parameters(shsq_sweep_set.v,
                                                                   shsq_sweep_set.t,
                                                                   1.02, 1.021)
# Build the extractors (we know stimulus starts at 1.02 s)
start = 1.02
spx = SpikeExtractor(start=start, dv_cutoff=dv_cutoff, thresh_frac=thresh_frac)
sptrx = SpikeTrainFeatureExtractor(start=start, end=None)

# Run the analysis
shsq_analysis = ShortSquareAnalysis(spx, sptrx)
results = shsq_analysis.analyze(shsq_sweep_set)

# Plot the sweeps at the lowest amplitude that evoked the most spikes
for i, swp in enumerate(shsq_sweep_set.sweeps):
    if i in results["common_amp_sweeps"].index:
        plt.plot(swp.t, swp.v, linewidth=0.5, color="steelblue")

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(1.015, 1.05)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")
