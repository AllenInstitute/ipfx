"""
Long Square Analysis
====================

Detect Long Square Features
"""

from ipfx.aibs_data_set import AibsDataSet
import ipfx.ephys_features as ft
import ipfx.ephys_extractor as efex
import ipfx.stimulus_protocol_analysis as spa

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
lsq_table = data_set.filtered_sweep_table(stimuli=data_set.long_square_names)
lsq_set = data_set.sweep_set(lsq_table.sweep_number)

# build the extractors
spx = efex.SpikeExtractor(start=1.02, end=2.02)
spfx = efex.SpikeTrainFeatureExtractor(start=1.02, end=2.02)

# run the analysis and print out a few of the features
lsqa = spa.LongSquareAnalysis(spx, spfx, subthresh_min_amp=-100.0)
data = lsqa.analyze(lsq_set)

for field in [ 'tau', 'v_baseline', 'input_resistance', 'vm_for_sag', 'fi_fit_slope', 'sag', 'rheobase_i' ]:
    print("%s: %s" % (field, str(data[field])))

# plot stim amp vs. firing rate
spiking_sweeps = data['spiking_sweeps'].sort_values(by='stim_amp')
plt.plot(spiking_sweeps.stim_amp,
         spiking_sweeps.avg_rate)
plt.xlabel('Stimulus amplitude (pA)')
plt.ylabel('Average firing rate (Hz)')
plt.show()





