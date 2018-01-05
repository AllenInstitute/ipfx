#!/usr/bin/python
import sys, logging
import os
import json
import shutil
import copy
import numpy as np
import shutil
import pandas as pd

import argschema as ags

import aibs.ipfx.data_set_features as dsfx
from aibs.ipfx.ephys_data_set import EphysDataSet
from aibs.ipfx._schemas import FeatureExtractionParameters
import aibs.ipfx.plot_qc_figures as plotqc

from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju

class PipelineDataSet(EphysDataSet):
    def __init__(self, sweep_list, file_name, ontology=None):
        super(PipelineDataSet, self).__init__(ontology)
        self.sweep_list = sweep_list
        self.sweep_table = pd.DataFrame.from_records(self.sweep_list)
        self.data_set = NwbDataSet(file_name)

    def sweep(self, sweep_number):
        return self.data_set.get_sweep(sweep_number)

def embed_spike_times(input_nwb_file, output_nwb_file, sweep_features):
    # embed spike times in NWB file
    logging.debug("Embedding spike times")
    tmp_nwb_file = output_nwb_file + ".tmp"

    shutil.copy(input_nwb_file, tmp_nwb_file)
    for sweep_num in sweep_features:
        spikes = sweep_features[sweep_num]['spikes']
        spike_times = [ s['threshold_t'] for s in spikes ]
        NwbDataSet(tmp_nwb_file).set_spike_times(sweep_num, spike_times)

    try:
        shutil.move(tmp_nwb_file, output_nwb_file)
    except OSError as e:
        logging.error("Problem renaming file: %s -> %s" % (tmp_nwb_file, output_nwb_file))
        raise e

def save_qc_figures(qc_fig_dir, data_set, feature_data, plot_cell_figures):
    if os.path.exists(qc_fig_dir):
        logging.warning("removing existing qc figures directory: %s", qc_fig_dir)
        shutil.rmtree(qc_fig_dir)

    Manifest.safe_mkdir(qc_fig_dir)

    logging.debug("saving qc plot figures")
    sweep_page = plotqc.make_sweep_page(data_set, feature_data, qc_fig_dir)
    plotqc.make_cell_page(data_set, feature_data, qc_fig_dir, save_cell_plots=plot_cell_figures)
            

def run_feature_extraction(input_nwb_file, output_nwb_file, qc_fig_dir, sweep_list, cell_features):
    input_cell_features = cell_features

    data_set = PipelineDataSet(sweep_list, input_nwb_file)
    
    cell_features, sweep_features, cell_record, sweep_records = dsfx.extract_data_set_features(data_set)

    if input_cell_features:
        cell_record.update(input_cell_features)

    feature_data = { 'cell_features': cell_features,
                     'sweep_features': sweep_features,
                     'cell_record': cell_record,
                     'sweep_records': sweep_records }

    embed_spike_times(input_nwb_file, output_nwb_file, sweep_features)
    save_qc_figures(qc_fig_dir, data_set, feature_data, True)

    return feature_data

def main():
    module = ags.ArgSchemaParser(schema_type=FeatureExtractionParameters)

    feature_data = run_feature_extraction(module.args["input_nwb_file"], 
                                          module.args["output_nwb_file"], 
                                          module.args["qc_fig_dir"], 
                                          module.args["sweep_list"], 
                                          module.args["cell_features"])

    ju.write(module.args["output_json"], feature_data)
            
if __name__ == "__main__": main()
