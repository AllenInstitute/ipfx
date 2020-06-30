"""
All Analysis
============

Run analysis for all stimulus types on data in the NWB file
"""
import os

from ipfx.dataset.create import create_ephys_data_set
from ipfx.data_set_features import extract_data_set_features
from ipfx.utilities import drop_failed_sweeps

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

# Calculate ephys features
cell_features, sweep_features, cell_record, sweep_records, _, _ = \
    extract_data_set_features(data_set, subthresh_min_amp=-100.0)

print(cell_record)
