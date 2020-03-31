"""
All Analysis
============

Run all analyses on NWB file
"""
import os

from allensdk.api.queries.cell_types_api import CellTypesApi
from ipfx.data_set_utils import create_data_set
from ipfx.data_set_features import extract_data_set_features

# Download and access the experimental data
ct = CellTypesApi()
nwb_file = os.path.join(
    os.path.dirname(os.getcwd()), 
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)
specimen_id = 595570553
sweep_info = ct.get_ephys_sweeps(specimen_id)
sweep_info = [
    sweep for sweep in sweep_info 
    if sweep["stimulus_name"] != "Test"
]

# remove test sweeps
sweep_info = [
    sweep for sweep in sweep_info 
    if sweep["stimulus_name"] != "Test"
]

data_set = create_data_set(sweep_info=sweep_info, nwb_file=nwb_file)

cell_features, sweep_features, cell_record, sweep_records = \
    extract_data_set_features(data_set, subthresh_min_amp=-100.0)

print(cell_record)
