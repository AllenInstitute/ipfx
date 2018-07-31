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

import allensdk.ipfx.data_set_features as dsft
from allensdk.ipfx.ephys_data_set import EphysDataSet, StimulusOntology
from allensdk.ipfx._schemas import FeatureExtractionParameters
import allensdk.ipfx.plot_qc_figures as plotqc
from allensdk.ipfx.aibs_data_set import AibsDataSet

from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju
from allensdk.core.nwb_data_set import NwbDataSet


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


def run_feature_extraction(input_nwb_file, stimulus_ontology_file, output_nwb_file, qc_fig_dir, sweep_props, cell_props):

    ont = StimulusOntology(ju.read(stimulus_ontology_file)) if stimulus_ontology_file else None
    data_set = AibsDataSet(sweep_props=sweep_props,
                           nwb_file=input_nwb_file,
                           ontology=ont,
                           api_sweeps=False)

    cell_features, sweep_features, cell_record, sweep_records = dsft.extract_data_set_features(data_set)

    if cell_props:
        cell_record.update(cell_props)

    feature_data = { 'cell_features': cell_features,
                     'sweep_features': sweep_features,
                     'cell_record': cell_record,
                     'sweep_records': sweep_records }

    embed_spike_times(input_nwb_file, output_nwb_file, sweep_features)
#    save_qc_figures(qc_fig_dir, data_set, feature_data, True)

    return feature_data


def main():
    module = ags.ArgSchemaParser(schema_type=FeatureExtractionParameters)

    feature_data = run_feature_extraction(module.args["input_nwb_file"],
                                          module.args.get("stimulus_ontology_file", None),
                                          module.args["output_nwb_file"],
                                          module.args["qc_fig_dir"],
                                          module.args["sweep_props"],
                                          module.args["cell_features"])

    ju.write(module.args["output_json"], feature_data)

if __name__ == "__main__": main()
