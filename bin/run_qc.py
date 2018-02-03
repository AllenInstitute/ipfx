#!/usr/bin/python
import logging
import sys
import math
import os
import re
import copy
import json
import numpy as np
import argparse
import h5py

from allensdk.ipfx.mies_nwb.mies_data_set import MiesDataSet
import allensdk.ipfx.qc_features as qcf

import argschema as ags
from allensdk.ipfx._schemas import QcParameters
import allensdk.core.json_utilities as ju

def run_qc(input_nwb_file, input_h5_file, cell_features, sweep_data, qc_criteria):
    ds = MiesDataSet(input_nwb_file, input_h5_file)
    cell_state, sweep_states = qcf.qc_experiment(ds, 
                                                 cell_features, 
                                                 sweep_data, 
                                                 qc_criteria)

    return dict(cell_state=cell_state, sweep_states=sweep_states)

def main():
    module = ags.ArgSchemaParser(schema_type=QcParameters)

    output = run_qc(module.args["input_nwb_file"],
                    module.args.get("input_h5_file"),                    
                    module.args["cell_features"],
                    module.args["sweep_data"],
                    module.args["qc_criteria"])

    ju.write(module.args["output_json"], output)


if __name__ == "__main__": main()
