"""
Long Square Analysis
====================

Calculate Features of Long Square sweeps
"""

from ipfx.feature_extractor import (
    SpikeFeatureExtractor, SpikeTrainFeatureExtractor
)
import ipfx.stimulus_protocol_analysis as spa
from ipfx.epochs import get_stim_epoch
from ipfx.dataset.create import create_ephys_data_set
from ipfx.utilities import drop_failed_sweeps

import os
import matplotlib.pyplot as plt

# Download and access the experimental data from DANDI archive per instructions in the documentation
# Example below will use an nwb file provided with the package

nwb_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)
# Create Ephys Data Set
data_set = create_ephys_data_set(nwb_file=nwb_file)

# Drop failed sweeps: sweeps with incomplete recording or failing QC criteria
drop_failed_sweeps(data_set)

# get sweep table of Long Square sweeps
long_square_table = data_set.filtered_sweep_table(
    stimuli=data_set.ontology.LONG_SQUARE_NAMES
)
long_square_sweeps = data_set.sweep_set(long_square_table.sweep_number)

# Select epoch corresponding to the actual recording from the sweeps
# and align sweeps so that the experiment would start at the same time
long_square_sweeps.select_epoch("recording")
long_square_sweeps.align_to_start_of_epoch("experiment")

# find the start and end time of the stimulus
# (treating the first sweep as representative)
stim_start_index, stim_end_index = get_stim_epoch(long_square_sweeps.i[0])
stim_start_time = long_square_sweeps.t[0][stim_start_index]
stim_end_time = long_square_sweeps.t[0][stim_end_index]

# build the extractors
spfx = SpikeFeatureExtractor(start=stim_start_time, end=stim_end_time)
sptfx = SpikeTrainFeatureExtractor(start=stim_start_time, end=stim_end_time)

# run the analysis and print out a few of the features
long_square_analysis = spa.LongSquareAnalysis(spfx, sptfx, subthresh_min_amp=-100.0)
data = long_square_analysis.analyze(long_square_sweeps)

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

# Plot stimulus amplitude vs. firing rate
spiking_sweeps = data['spiking_sweeps'].sort_values(by='stim_amp')
plt.plot(spiking_sweeps.stim_amp,
         spiking_sweeps.avg_rate)
plt.xlabel('Stimulus amplitude (pA)')
plt.ylabel('Average firing rate (Hz)')

plt.show()
