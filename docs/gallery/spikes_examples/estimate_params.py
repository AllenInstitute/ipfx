"""
Estimate Spike Detection Parameters
===================================

Estimate spike detection parameters
"""

import os
from ipfx.data_set_utils import create_data_set
from ipfx.spike_features import estimate_adjusted_detection_parameters
from ipfx.feature_extractor import SpikeFeatureExtractor
from ipfx.utilities import drop_failed_sweeps
import matplotlib.pyplot as plt

# Download and access the experimental data from DANDI archive per instructions in the documentation
# Example below will use an nwb file provided with the package

nwb_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)

# Create data set from the nwb file and find the short squares
data_set = create_data_set(nwb_file=nwb_file)

# Drop failed sweeps: sweeps with incomplete recording or failing QC criteria
drop_failed_sweeps(data_set)

short_square_table = data_set.filtered_sweep_table(stimuli=["Short Square"])
ssq_set = data_set.sweep_set(short_square_table.sweep_number)

# estimate the dv cutoff and threshold fraction
dv_cutoff, thresh_frac = estimate_adjusted_detection_parameters(
    ssq_set.v, ssq_set.t, 1.02, 1.021
)

# detect spikes  in a given sweep number
sweep_number = 16
sweep = data_set.sweep(sweep_number)
ext = SpikeFeatureExtractor(dv_cutoff=dv_cutoff, thresh_frac=thresh_frac)
spikes = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)

# and plot them
plt.plot(sweep.t, sweep.v)
plt.plot(spikes["peak_t"], spikes["peak_v"], 'r.')
plt.plot(spikes["threshold_t"], spikes["threshold_v"], 'k.')
plt.xlim(1.018, 1.028)
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.show()
