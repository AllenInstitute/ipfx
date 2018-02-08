import logging
import sys
import numpy as np
import h5py

from allensdk.ipfx.ephys_data_set import EphysStimulusOntology
from allensdk.ipfx.mies_nwb.mies_data_set import MiesDataSet

import allensdk.ipfx.qc_features as qcf
import allensdk.core.json_utilities as ju

import argschema as ags
from allensdk.ipfx._schemas import SweepExtractionParameters

# manual keys are values that can be passed in through input.json.
# these values are used if the particular value cannot be computed.
# a better name might be 'DEFAULT_VALUE_KEYS'
MANUAL_KEYS = ['manual_seal_gohm', 'manual_initial_access_resistance_mohm', 'manual_initial_input_mohm' ]

def run_sweep_extraction(input_nwb_file, input_h5_file, stimulus_ontology_file, input_manual_values=None):
    if input_manual_values is None:
        input_manual_values = {}

    manual_values = {}
    for mk in MANUAL_KEYS:
        if mk in input_manual_values:
            manual_values[mk] = input_manual_values[mk]

    ont = EphysStimulusOntology(ju.read(stimulus_ontology_file))
    ds = MiesDataSet(input_nwb_file, input_h5_file, ontology=ont)
    cell_features, cell_tags = qcf.cell_qc_features(ds, manual_values)
    sweep_features = qcf.sweep_qc_features(ds)
    
    return dict(cell_features=cell_features,
                cell_tags=cell_tags,
                sweep_data=sweep_features)

def main():
    module = ags.ArgSchemaParser(schema_type=SweepExtractionParameters)    
    output = run_sweep_extraction(module.args["input_nwb_file"],
                                  module.args.get("input_h5_file"),
                                  module.args["stimulus_ontology_file"])
    ju.write(module.args["output_json"], output)

if __name__ == "__main__": main()
