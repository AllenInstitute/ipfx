import logging
import sys
import numpy as np
import h5py

from aibs.ipfx.ephys_data_set import EphysStimulusOntology
from aibs.ipfx.mies_nwb.mies_data_set import MiesDataSet

import aibs.ipfx.qc_features as qcf
import allensdk.core.json_utilities as ju

import argschema as ags
from aibs.ipfx._schemas import SweepExtractionParameters

# manual keys are values that can be passed in through input.json.
# these values are used if the particular value cannot be computed.
# a better name might be 'DEFAULT_VALUE_KEYS'
MANUAL_KEYS = ['manual_seal_gohm', 'manual_initial_access_resistance_mohm', 'manual_initial_input_mohm' ]

def main():
    module = ags.ArgSchemaParser(schema_type=SweepExtractionParameters)    
    args = module.args

    nwb_file = args["input_nwb_file"]
    manual_values = {}
    for mk in MANUAL_KEYS:
        if mk in args:
            manual_values[mk] = args[mk]

    ont = EphysStimulusOntology(args['stimulus_ontology_file'])
    ds = MiesDataSet(nwb_file, ontology=ont)
    cell_features, cell_tags = qcf.cell_qc_features(ds, manual_values)
    sweep_features = qcf.sweep_qc_features(ds)

    ju.write(args["output_json"], dict(cell_features=cell_features,
                                       cell_tags=cell_tags,
                                       sweep_data=sweep_features))

if __name__ == "__main__": main()
