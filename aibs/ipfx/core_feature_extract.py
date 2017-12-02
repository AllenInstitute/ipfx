import sys, os, shutil
import logging
from collections import defaultdict
import numpy as np
import json

from allensdk.config.manifest import Manifest
from allensdk.core.json_utilities import json_handler

from allensdk.core.nwb_data_set import NwbDataSet
from allensdk.ephys.extract_cell_features import extract_cell_features, extract_sweep_features
from allensdk.ephys.ephys_features import FeatureError
from allensdk.ephys.ephys_extractor import reset_long_squares_start
import allensdk.internal.ephys.plot_qc_figures as plot_qc_figures


TEST_PULSE_DURATION_SEC = 0.4

LONG_SQUARE_COARSE = 'C1LSCOARSE'
LONG_SQUARE_FINE = 'C1LSFINEST'
SHORT_SQUARE = 'C1SSFINEST'
RAMP = 'C1RP25PR1S'
PASSED_SWEEP_STATES = [ 'manual_passed', 'auto_passed' ]
ICLAMP_UNITS = [ 'Amps', 'pA' ]

def save_qc_figures(qc_fig_dir, nwb_file, output_data, plot_cell_figures):
    if os.path.exists(qc_fig_dir):
        logging.warning("removing existing qc figures directory: %s", qc_fig_dir)
        shutil.rmtree(qc_fig_dir)

    Manifest.safe_mkdir(qc_fig_dir)

    logging.debug("saving qc plot figures")
    plot_qc_figures.make_sweep_page(nwb_file, output_data, qc_fig_dir)
    plot_qc_figures.make_cell_page(nwb_file, output_data, qc_fig_dir, plot_cell_figures)
