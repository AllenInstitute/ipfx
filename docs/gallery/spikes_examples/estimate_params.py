"""
Estimate Spike Detection Parameters
===================================

Estimate spike detection parameters
"""

from ipfx.data_set_utils import create_data_set
from ipfx.spike_features import estimate_adjusted_detection_parameters
from ipfx.feature_extractor import SpikeFeatureExtractor

from allensdk.api.queries.cell_types_api import CellTypesApi

import os
import matplotlib.pyplot as plt

# Download and access the experimental data
ct = CellTypesApi()
specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
sweep_info = ct.get_ephys_sweeps(specimen_id)

if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)

# build a data set and find the short squares
data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)
ssq_table = data_set.filtered_sweep_table(stimuli=["Short Square"])
ssq_set = data_set.sweep_set(ssq_table.sweep_number)

# estimate the dv cutoff and threshold fraction
dv_cutoff, thresh_frac = estimate_adjusted_detection_parameters(
    ssq_set.v, ssq_set.t, 1.02, 1.021
)

# detect spikes
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
