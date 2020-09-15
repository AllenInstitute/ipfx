import logging
import argschema as ags

import allensdk.core.json_utilities as json_utilities

import ipfx.sweep_props as sweep_props
from ipfx.logging_utils import log_pretty_header
from ipfx._schemas import PipelineParameters

from ipfx.bin.run_sweep_extraction import run_sweep_extraction
from ipfx.bin.run_qc import run_qc
from ipfx.bin.run_feature_extraction import run_feature_extraction


def run_pipeline(
        input_nwb_file,
        output_nwb_file,
        stimulus_ontology_file,
        qc_fig_dir,
        qc_criteria,
        manual_sweep_states,
        write_spikes=True,
        update_ontology=True
):

    se_output = run_sweep_extraction(
        input_nwb_file,
        stimulus_ontology_file,
        update_ontology=update_ontology
    )

    sweep_props.drop_tagged_sweeps(se_output["sweep_features"])
    sweep_props.remove_sweep_feature("tags", se_output["sweep_features"])

    qc_output = run_qc(
        stimulus_ontology_file,
        se_output["cell_features"],
        se_output["sweep_features"],
        qc_criteria
    )

    sweep_props.override_auto_sweep_states(
        manual_sweep_states, qc_output["sweep_states"]
    )
    sweep_props.assign_sweep_states(
        qc_output["sweep_states"], se_output["sweep_features"]
    )

    fx_output = run_feature_extraction(
        input_nwb_file,
        stimulus_ontology_file,
        output_nwb_file,
        qc_fig_dir,
        se_output['sweep_features'],
        se_output['cell_features'],
        write_spikes,
    )

    log_pretty_header("Analysis completed!", level=1)

    return {
        "sweep_extraction": se_output,
        "qc": qc_output,
        "feature_extraction": fx_output
    }


def main():

    """
    Usage:
    python run_pipeline.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=PipelineParameters)

    output = run_pipeline(
        module.args["input_nwb_file"],
        module.args["output_nwb_file"],
        module.args.get("stimulus_ontology_file", None),
        module.args.get("qc_fig_dir", None),
        module.args.get("qc_criteria", None),
        module.args.get("manual_sweep_states", None),
        module.args.get("write_spikes", None),
        module.args["update_stimulus_ontology"],
    )

    json_utilities.write(module.args["output_json"], output)


if __name__ == "__main__":
    main()
