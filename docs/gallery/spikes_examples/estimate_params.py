"""
Estimate Spike Detection Parameters
===================================

Estimate spike detection parameters
"""

from ipfx.aibs_data_set import AibsDataSet
import ipfx.ephys_features as ft
import ipfx.ephys_extractor as fx

from allensdk.api.queries.cell_types_api import CellTypesApi

import os
import matplotlib.pyplot as plt

specimen_id = 595570553
nwb_file = '%d.nwb' % specimen_id

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweeps = ct.get_ephys_sweeps(specimen_id)

# build a data set and find the short squares
data_set = AibsDataSet(sweeps, nwb_file)
ssq_table = data_set.filtered_sweep_table(stimuli=["Short Square"])
ssq_set = data_set.sweep_set(ssq_table.sweep_number)

# estimate the dv cutoff and threshold fraction
dv_cutoff, thresh_frac = ft.estimate_adjusted_detection_parameters(ssq_set.v,
                                                                   ssq_set.t,
                                                                   1.02, 1.021)

# detect spikes
sweep_number = 16
sweep = data_set.sweep(sweep_number)
ext = fx.SpikeExtractor(dv_cutoff=dv_cutoff, thresh_frac=thresh_frac)
spikes = ext.process(t=sweep.t, v=sweep.v, i=sweep.i)


# and plot them
plt.plot(sweep.t, sweep.v)
plt.plot(spikes["peak_t"], spikes["peak_v"], 'r.')
plt.plot(spikes["threshold_t"], spikes["threshold_v"], 'k.')
plt.xlim(1.018,1.028)
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)')
plt.show()
