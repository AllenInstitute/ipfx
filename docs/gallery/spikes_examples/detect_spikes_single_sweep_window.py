"""
Single sweep detection with analysis window
===========================================

Detect spikes for a single sweep in a specified time window
"""

import os
import matplotlib.pyplot as plt
from ipfx.dataset.create import create_ephys_data_set
from ipfx.feature_extractor import SpikeFeatureExtractor

# Download and access the experimental data from DANDI archive per instructions in the documentation
# Example below will use an nwb file provided with the package

nwb_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)

# Create data set from the nwb file and choose a sweeep
dataset = create_ephys_data_set(nwb_file=nwb_file)
sweep = dataset.sweep(sweep_number=39)

# Configure the extractor to just detect spikes in the middle of the step
ext = SpikeFeatureExtractor(start=1.25, end=1.75)
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

plt.show()

