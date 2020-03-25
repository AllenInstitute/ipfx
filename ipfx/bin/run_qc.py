#!/usr/bin/python
import logging
from ipfx.stimulus import StimulusOntology
import ipfx.qc_feature_evaluator as qcp
import argschema as ags
from ipfx._schemas import QcParameters
import allensdk.core.json_utilities as ju
import ipfx.sweep_props as sp
import pandas as pd
import ipfx.logging_utils as lu


def run_qc(stimulus_ontology_file, cell_features, sweep_features, qc_criteria):
    """

    Parameters
    ----------
    stimulus_ontology_file : str
        ontology file name
    cell_features: dict
        cell features
    sweep_features : list of dicts
        sweep features
    qc_criteria: dict
        qc criteria

    Returns
    -------
    dict
        containing state of the cell and sweeps
    """

    lu.log_pretty_header("Perform QC checks", level=1)

    if not stimulus_ontology_file:
        stimulus_ontology_file = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
        logging.info(F"Ontology is not provided, using default {StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE}")

    ont = StimulusOntology(ju.read(stimulus_ontology_file))

    cell_state, sweep_states = qcp.qc_experiment(ont,
                                                 cell_features,
                                                 sweep_features,
                                                 qc_criteria)


    qc_summary(sweep_features, sweep_states, cell_features, cell_state)

    return dict(cell_state=cell_state, sweep_states=sweep_states)


def qc_summary(sweep_features, sweep_states, cell_features, cell_state):
    """
    Output QC summary

    Parameters
    ----------
    sweep_features: list of dicts
    sweep_states: list of dict
    cell_features: list of dicts
    cell_state: dict

    Returns
    -------

    """
    lu.log_pretty_header("QC Summary:",level=2)

    logging.info("Cell State:")
    for k,v in cell_state.items():
        logging.info("%s:%s" % (k,v))

    logging.info("Sweep States:")

    sp.assign_sweep_states(sweep_states, sweep_features)
    sweep_table = pd.DataFrame(sweep_features)

    if sweep_features:
        for stimulus_name, sg_table in sweep_table.groupby("stimulus_name"):
            passed_sweep_numbers = sg_table[sg_table.passed == True].sweep_number.sort_values().values
            failed_sweep_numbers = sg_table[sg_table.passed == False].sweep_number.sort_values().values

            logging.info("{} sweeps passed: {}, failed {}".format(stimulus_name, passed_sweep_numbers,failed_sweep_numbers))
    else:
        logging.warning("No current clamp sweeps available for QC")



def main():
    """
    Usage:
    python run_qc.py --input_json INPUT_JSON --output_json OUTPUT_JSON


    """
    module = ags.ArgSchemaParser(schema_type=QcParameters)

    output = run_qc(module.args.get("stimulus_ontology_file", None),
                    module.args["cell_features"],
                    module.args["sweep_features"],
                    module.args["qc_criteria"])

    ju.write(module.args["output_json"], output)


if __name__ == "__main__": main()
