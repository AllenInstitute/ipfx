import logging
from ipfx.stimulus import StimulusOntology
import ipfx.qc_feature_extractor as qcfe
import allensdk.core.json_utilities as ju

import argschema as ags
from ipfx._schemas import SweepExtractionParameters
from ipfx.data_set_utils import create_data_set
import ipfx.logging_utils as lu

# manual keys are values that can be passed in through input.json.
# these values are used if the particular value cannot be computed.
# a better name might be 'DEFAULT_VALUE_KEYS'
MANUAL_KEYS = ['manual_seal_gohm', 'manual_initial_access_resistance_mohm', 'manual_initial_input_mohm' ]


def run_sweep_extraction(input_nwb_file, input_h5_file, stimulus_ontology_file, input_manual_values=None):
    """
    Parameters
    ----------
    input_nwb_file
    input_h5_file
    stimulus_ontology_file
    input_manual_values

    Returns
    -------

    """
    lu.log_pretty_header("Extract QC features", level=1)

    if input_manual_values is None:
        input_manual_values = {}

    manual_values = {}
    for mk in MANUAL_KEYS:
        if mk in input_manual_values:
            manual_values[mk] = input_manual_values[mk]

    ont = StimulusOntology(ju.read(stimulus_ontology_file)) if stimulus_ontology_file else StimulusOntology()
    ds = create_data_set(nwb_file=input_nwb_file,
                         h5_file=input_h5_file,
                         ontology=ont)

    cell_features, cell_tags = qcfe.cell_qc_features(ds, manual_values)

    for tag in cell_tags:
        logging.warning(tag)

    sweep_features = qcfe.sweep_qc_features(ds)

    return dict(cell_features=cell_features,
                cell_tags=cell_tags,
                sweep_features=sweep_features,
                )


def main():
    """
    Usage:
    python run_sweep_extraction.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=SweepExtractionParameters)
    output = run_sweep_extraction(module.args["input_nwb_file"],
                                  module.args.get("input_h5_file",None),
                                  module.args.get("stimulus_ontology_file", None))

    ju.write(module.args["output_json"], output)

if __name__ == "__main__": main()
