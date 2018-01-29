"""
Single sweep detection
=================================

Detect spikes for a single sweep
"""


# Download and access the experimental data
from allensdk.api.queries.cell_types_api import CellTypesApi

ct = CellTypesApi()

specimen_id = 488679042
nwb_filename = "example.nwb"
ct.save_ephys_data(specimen_id, nwb_filename)


# Get the data for the sweep into a format we can use
from aibs.ipfx.aibs_data_set import AibsDataSet

dataset = AibsDataSet([], nwb_filename)
sweep_number = 60
sweep = dataset.sweep(sweep_number)

# Extract information about the spikes
from aibs.ipfx.ephys_extractor import SpikeExtractor

ext = SpikeExtractor()
results = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)

# Plot the results, showing two features of the detected spikes
import matplotlib.pyplot as plt

plt.plot(sweep.t, sweep.v)
plt.plot(results["peak_t"], results["peak_v"], 'r.')
plt.plot(results["threshold_t"], results["threshold_v"], 'k.')

# Set the plot limits to highlight where spikes are and set axis labels
plt.xlim(0.5, 2.5)
plt.xlabel("Time (s)")
plt.ylabel("Membrane potential (mV)")

##################################
# Link to the example data used here: http://celltypes.brain-map.org/experiment/electrophysiology/488679042
