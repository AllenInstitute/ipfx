#!/usr/bin/python
import logging

from ipfx.stimulus import StimulusOntology
import ipfx.qc_protocol as qcp

import argschema as ags
from ipfx._schemas import QcParameters
import allensdk.core.json_utilities as ju


def run_qc(stimulus_ontology_file, cell_features, sweep_features, qc_criteria):
    """
    Usage:
    python run_qc.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    Run example:
    python run_qc.py --input_json ../../tests/module_io/Ephys_Roi_Result_730744302/qc_input.json --output_json ../../tests/module_io/Ephys_Roi_Result_730744302/qc_output.json


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

    logging.debug("stimulus ontology file: %s", stimulus_ontology_file)
    ont = StimulusOntology(ju.read(stimulus_ontology_file)) if stimulus_ontology_file else StimulusOntology()

    cell_state, sweep_states = qcp.qc_experiment(ont,
                                                 cell_features,
                                                 sweep_features,
                                                 qc_criteria)

    return dict(cell_state=cell_state, sweep_states=sweep_states)


def main():
    module = ags.ArgSchemaParser(schema_type=QcParameters)

    output = run_qc(module.args.get("stimulus_ontology_file", None),
                    module.args["cell_features"],
                    module.args["sweep_features"],
                    module.args["qc_criteria"])

    ju.write(module.args["output_json"], output)


if __name__ == "__main__": main()
