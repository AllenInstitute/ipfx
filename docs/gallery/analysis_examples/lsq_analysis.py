"""
Long Square Analysis
====================

Detect Long Square Features
"""
from ipfx.aibs_data_set import AibsDataSet
from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)
import ipfx.stimulus_protocol_analysis as spa
from ipfx.epochs import get_stim_epoch

from allensdk.api.queries.cell_types_api import CellTypesApi

import os
import matplotlib.pyplot as plt

# download a specific experiment NWB file via AllenSDK
ct = CellTypesApi()

specimen_id = 595570553
nwb_file = "%d.nwb" % specimen_id
if not os.path.exists(nwb_file):
    ct.save_ephys_data(specimen_id, nwb_file)
sweep_info = ct.get_ephys_sweeps(specimen_id)

# build a data set and find the long squares
data_set = AibsDataSet(sweep_info=sweep_info, nwb_file=nwb_file)
lsq_table = data_set.filtered_sweep_table(
    stimuli=data_set.ontology.long_square_names
)
lsq_set = data_set.sweep_set(lsq_table.sweep_number)

# find the start and end time of the stimulus 
# (in this case, this is the same across long-square sweeps, 
# so we'll just use the first)
stim_start_index, stim_end_index = get_stim_epoch(lsq_set.i[0])
stim_start_time = lsq_set.t[0][stim_start_index]
stim_end_time = lsq_set.t[0][stim_end_index]

# build the extractors
spx = SpikeFeatureExtractor(start=stim_start_time, end=stim_end_time)
spfx = SpikeTrainFeatureExtractor(start=stim_start_time, end=stim_end_time)

# run the analysis and print out a few of the features
lsqa = spa.LongSquareAnalysis(spx, spfx, subthresh_min_amp=-100.0)
data = lsqa.analyze(lsq_set)

fields_to_print = [
    'tau', 
    'v_baseline', 
    'input_resistance', 
    'vm_for_sag', 
    'fi_fit_slope', 
    'sag', 
    'rheobase_i'
]

for field in fields_to_print:
    print("%s: %s" % (field, str(data[field])))

# plot stim amp vs. firing rate
spiking_sweeps = data['spiking_sweeps'].sort_values(by='stim_amp')
plt.plot(spiking_sweeps.stim_amp,
         spiking_sweeps.avg_rate)
plt.xlabel('Stimulus amplitude (pA)')
plt.ylabel('Average firing rate (Hz)')

plt.show()

