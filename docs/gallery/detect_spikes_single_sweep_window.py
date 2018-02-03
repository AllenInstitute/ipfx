"""
Single sweep detection with analysis window
=================================

Detect spikes for a single sweep in a specified time window
"""

import os
import matplotlib.pyplot as plt
from allensdk.api.queries.cell_types_api import CellTypesApi
from allensdk.ipfx.aibs_data_set import AibsDataSet
from allensdk.ipfx.ephys_extractor import SpikeExtractor

# Download and access the experimental data
ct = CellTypesApi()

specimen_id = 488679042
nwb_filename = "%d.nwb" % specimen_id
if not os.path.exists(nwb_filename):
    ct.save_ephys_data(specimen_id, nwb_filename)

# Get the data for the sweep into a format we can use
dataset = AibsDataSet([], nwb_filename)
sweep_number = 60
sweep = dataset.sweep(sweep_number)

# Configure the extractor to just detect spikes in the middle of the step
ext = SpikeExtractor(start=1.25, end=1.75)
results = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)

# Plot the results, showing two features of the detected spikes
plt.plot(sweep.t, sweep.v)
plt.plot(results["peak_t"], results["peak_v"], 'r.')
plt.plot(results["threshold_t"], results["threshold_v"], 'k.')

# Set the plot limits to highlight where spikes are and axis labels
plt.xlim(0.5, 2.5)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")

# Show the analysis window on the plot
plt.axvline(1.25, linestyle="dotted", color="gray")
plt.axvline(1.75, linestyle="dotted", color="gray")


##################################
# Link to the example data used here: http://celltypes.brain-map.org/experiment/electrophysiology/488679042
