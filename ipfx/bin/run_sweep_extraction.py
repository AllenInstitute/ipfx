import logging

import allensdk.core.json_utilities as json_utilities
import argschema as ags

from ipfx._schemas import SweepExtractionParameters
from ipfx.data_set_utils import create_data_set
from ipfx.logging_utils import log_pretty_header
from ipfx.bin.make_stimulus_ontology import make_stimulus_ontology_from_lims
from ipfx.stimulus import StimulusOntology
from ipfx.qc_feature_extractor import cell_qc_features, sweep_qc_features

# manual keys are values that can be passed in through input.json.
# these values are used if the particular value cannot be computed.
# a better name might be 'DEFAULT_VALUE_KEYS'
MANUAL_KEYS = (
    'manual_seal_gohm',
    'manual_initial_access_resistance_mohm',
    'manual_initial_input_mohm'
)


def run_sweep_extraction(
        input_nwb_file,
        stimulus_ontology_file,
        input_manual_values=None
):
    """
    Parameters
    ----------
    input_nwb_file
    stimulus_ontology_file
    input_manual_values

    Returns
    -------
    """
    log_pretty_header("Extract QC features", level=1)

    if input_manual_values is None:
        input_manual_values = {}

    manual_values = {}
    for mk in MANUAL_KEYS:
        if mk in input_manual_values:
            manual_values[mk] = input_manual_values[mk]

    if stimulus_ontology_file:
        make_stimulus_ontology_from_lims(stimulus_ontology_file)
    else:
        stimulus_ontology_file = \
            StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
        logging.info(
            f"Ontology is not provided, using default "
            f"{StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE}"
        )

    ont = StimulusOntology(json_utilities.read(stimulus_ontology_file))
    ds = create_data_set(
        nwb_file=input_nwb_file,
        ontology=ont
    )

    cell_features, cell_tags = cell_qc_features(ds, manual_values)

    for tag in cell_tags:
        logging.warning(tag)

    sweep_features = sweep_qc_features(ds)

    return {
        "cell_features": cell_features,
        "cell_tags": cell_tags,
        "sweep_features": sweep_features,
    }


def main():
    """
    Usage:
    python run_sweep_extraction.py
        --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=SweepExtractionParameters)
    output = run_sweep_extraction(
        module.args["input_nwb_file"],
        module.args.get("stimulus_ontology_file", None)
    )

    json_utilities.write(module.args["output_json"], output)


if __name__ == "__main__":
    main()
