"""
Sweep QC Features
=================

Estimate sweep QC features
"""
import os
import pandas as pd
from ipfx.dataset.create import create_ephys_data_set
from ipfx.qc_feature_extractor import sweep_qc_features

import ipfx.sweep_props as sweep_props
import ipfx.qc_feature_evaluator as qcp
from ipfx.stimulus import StimulusOntology


# Download and access the experimental data from DANDI archive per instructions in the documentation
# Example below will use an nwb file provided with the package

nwb_file = os.path.join(
    os.path.dirname(os.getcwd()),
    "data",
    "nwb2_H17.03.008.11.03.05.nwb"
)
data_set = create_ephys_data_set(nwb_file=nwb_file)

# Compute sweep QC features
sweep_features = sweep_qc_features(data_set)

# Drop sweeps that failed to compute QC criteria
sweep_props.drop_tagged_sweeps(sweep_features)
sweep_props.remove_sweep_feature("tags", sweep_features)

stimulus_ontology = StimulusOntology.default()
qc_criteria = qcp.load_default_qc_criteria()

sweep_states = qcp.qc_sweeps(
    stimulus_ontology, sweep_features, qc_criteria
)

# print a few sweeps and states
print(pd.DataFrame(sweep_features).head())
print(sweep_states[0:len(pd.DataFrame(sweep_features).head())])
