import os
import logging
import shutil

import allensdk.ipfx.plot_qc_figures as plotqc
from allensdk.ipfx.ephys_data_set import StimulusOntology
from allensdk.ipfx.aibs_data_set import AibsDataSet

from allensdk.config.manifest import Manifest
import allensdk.core.json_utilities as ju


def display_features(qc_fig_dir, data_set, feature_data, plot_sweep_figures=True, plot_cell_figures=True):
    if os.path.exists(qc_fig_dir):
        logging.warning("Removing existing qc figures directory: %s", qc_fig_dir)
        shutil.rmtree(qc_fig_dir)

    image_dir = os.path.join(qc_fig_dir,"img")
    Manifest.safe_mkdir(qc_fig_dir)
    Manifest.safe_mkdir(image_dir)

    logging.debug("Saving qc plot figures")
    if plot_sweep_figures:
        plotqc.make_sweep_page(data_set, qc_fig_dir)
    plotqc.make_cell_page(data_set, feature_data, qc_fig_dir, save_cell_plots=plot_cell_figures)


def run_visualization(input_nwb_file, stimulus_ontology_file, qc_fig_dir, sweep_info, feature_data):


    ont = StimulusOntology(ju.read(stimulus_ontology_file)) if stimulus_ontology_file else None
    data_set = AibsDataSet(sweep_info=sweep_info,
                           nwb_file=input_nwb_file,
                           ontology=ont,
                           api_sweeps=False)


    display_features(qc_fig_dir, data_set, feature_data, plot_sweep_figures=True, plot_cell_figures=False)
