#!/usr/bin/python
import logging
import shutil
import argschema as ags
import ipfx.error as er
import ipfx.data_set_features as dsft
from ipfx.stimulus import StimulusOntology
from ipfx._schemas import FeatureExtractionParameters
from ipfx.data_set_utils import create_data_set
import ipfx.sweep_props as sp

import allensdk.core.json_utilities as ju
from allensdk.core.nwb_data_set import NwbDataSet

import ipfx.plot_qc_figures as plotqc
import ipfx.logging_utils as lu


def embed_spike_times(input_nwb_file, output_nwb_file, sweep_features):
    # embed spike times in NWB file
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
    logging.debug("Embedded spike times into output.nwb file")


def run_feature_extraction(input_nwb_file,
                           stimulus_ontology_file,
                           output_nwb_file,
                           qc_fig_dir,
                           sweep_info,
                           cell_info):

    lu.log_pretty_header("Extract ephys features", level=1)

    sp.drop_failed_sweeps(sweep_info)
    if len(sweep_info) == 0:
        raise er.FeatureError("There are no QC-passed sweeps available to analyze")

    ont = StimulusOntology(ju.read(stimulus_ontology_file)) if stimulus_ontology_file else StimulusOntology()

    data_set = create_data_set(sweep_info=sweep_info,
                               nwb_file=input_nwb_file,
                               ontology=ont,
                               api_sweeps=False)

    try:
        cell_features, sweep_features, cell_record, sweep_records = dsft.extract_data_set_features(data_set)

        if cell_info: cell_record.update(cell_info)

        cell_state = {"failed_fx": False, "fail_fx_message": None}

        feature_data = { 'cell_features': cell_features,
                         'sweep_features': sweep_features,
                         'cell_record': cell_record,
                         'sweep_records': sweep_records,
                         'cell_state': cell_state
                         }

    except (er.FeatureError,IndexError) as e:
        cell_state = {"failed_fx":True, "fail_fx_message": e}
        logging.warning(e)
        feature_data = {'cell_state': cell_state}

    if not cell_state["failed_fx"]:
        embed_spike_times(input_nwb_file, output_nwb_file, sweep_features)

        if qc_fig_dir is None:
            logging.info("qc_fig_dir is not provided, will not save figures")
        else:
            plotqc.display_features(qc_fig_dir, data_set, feature_data)

        # On Windows int64 keys of sweep numbers cannot be converted to str by json.dump when serializing.
        # Thus, we are converting them here:
        feature_data["sweep_features"] = {str(k): v for k, v in feature_data["sweep_features"].items()}

    return feature_data


def main():
    """
    Usage:
    python run_feature_extraction.py --input_json INPUT_JSON --output_json OUTPUT_JSON

    """

    module = ags.ArgSchemaParser(schema_type=FeatureExtractionParameters)

    feature_data = run_feature_extraction(module.args["input_nwb_file"],
                                          module.args.get("stimulus_ontology_file", None),
                                          module.args["output_nwb_file"],
                                          module.args.get("qc_fig_dir", None),
                                          module.args["sweep_features"],
                                          module.args["cell_features"])

    ju.write(module.args["output_json"], feature_data)

if __name__ == "__main__": main()
