"""
Single sweep detection
=================================

Detect spikes for a single sweep
"""

import os
import matplotlib.pyplot as plt
from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor

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
ext = SpikeFeatureExtractor()
results = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)

# Plot the results, showing two features of the detected spikes
plt.plot(sweep.t, sweep.v)
plt.plot(results["peak_t"], results["peak_v"], 'r.')
plt.plot(results["threshold_t"], results["threshold_v"], 'k.')

# Set the plot limits to highlight where spikes are and set axis labels
plt.xlim(0.5, 2.5)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")

plt.show()

##################################
# Link to the example data used here: http://celltypes.brain-map.org/experiment/electrophysiology/488679042
