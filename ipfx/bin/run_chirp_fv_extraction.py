import numpy as np
import argschema as ags
from ipfx.aibs_data_set import AibsDataSet
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import h5py
import os
import json
import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
from ipfx.bin.run_feature_vector_extraction import lims_nwb_information, sdk_nwb_information
from ipfx.chirp import extract_chirp_feature_vector, ontology_with_chirps, CHIRP_CODES

class CollectChirpFeatureVectorParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile(
        description="input text file with specimen numbers of each line",
        default=None, allow_none=True)
    include_failed_cells = ags.fields.Boolean(
        description="whether to include failed cells",
        default=False)
    run_parallel = ags.fields.Boolean(
        description="whether to use multiprocessing",
        default=True)
    output_dir = ags.fields.OutputFile(
        description="output directory", default=None)
    output_code = ags.fields.String(
        description="output code for naming files", default=None)
    chirp_stimulus_codes = ags.fields.List(ags.fields.String,
        description="stimulus codes for chirps",
        default=CHIRP_CODES,
        cli_as_single_argument=True)
    data_source = ags.fields.String(
        description="Source of NWB files ('sdk' or 'lims')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims"]
        )

def data_for_specimen_id(specimen_id, data_source="lims", chirp_stimulus_codes=CHIRP_CODES):
    logging.debug("specimen_id: {}".format(specimen_id))
    ontology = ontology_with_chirps(chirp_stimulus_codes)
    # Find or retrieve NWB file and ancillary info and construct an AibsDataSet object
    if data_source == "lims":
        nwb_path, h5_path = lims_nwb_information(specimen_id)
        if type(nwb_path) is dict and "error" in nwb_path:
            logging.warning("Problem getting NWB file for specimen {:d} from LIMS".format(specimen_id))
            return nwb_path

        try:
            data_set = AibsDataSet(
                nwb_file=nwb_path, h5_file=h5_path, ontology=ontology)
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} from LIMS".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    elif data_source == "sdk":
        nwb_path, sweep_info = sdk_nwb_information(specimen_id)
        try:
            data_set = AibsDataSet(
                nwb_file=nwb_path, sweep_info=sweep_info, ontology=ontology)
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} via Allen SDK".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    else:
        logging.error("invalid data source specified ({})".format(data_source))


    # Identify chirp sweeps

    try:
        iclamp_st = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP, stimuli=["Chirp"])
        chirp_sweep_numbers = iclamp_st["sweep_number"].sort_values().values
    except Exception as detail:
        logging.warning("Exception when identifying sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=1)}}

    if len(chirp_sweep_numbers) == 0:
        logging.info("No chirp sweeps for {:d}".format(specimen_id))
        return {"error": {"type": "processing", "details:": "no available chirp sweeps"}}

    try:
        result = extract_chirp_feature_vector(data_set, chirp_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when processing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=1)}}

    return result


def run_chirp_feature_vector_extraction(output_dir, output_code, include_failed_cells,
        specimen_ids, chirp_stimulus_codes=CHIRP_CODES, data_source="lims", run_parallel=True):
    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    get_data_partial = partial(data_for_specimen_id,
                               data_source=data_source,
                               chirp_stimulus_codes=chirp_stimulus_codes)
    if run_parallel:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)
    filtered_set = [(i, r) for i, r in zip(specimen_ids, results) if not "error" in r.keys()]
    error_set = [{"id": i, "error": d} for i, d in zip(specimen_ids, results) if "error" in d.keys()]
    if len(filtered_set) == 0:
        logging.info("No specimens had results")
        return

    used_ids, results = zip(*filtered_set)
    logging.info("Finished with {:d} processed specimens".format(len(used_ids)))
    k_sizes = {}
    for k in results[0].keys():
        if k not in k_sizes and results[0][k] is not None:
            k_sizes[k] = len(results[0][k])
        data = np.array([r[k] if k in r else np.nan * np.zeros(k_sizes[k])
                        for r in results])
        if len(data) < len(used_ids):
            logging.warning("Missing data!")
            missing = np.array([k not in r for r in results])
            print(k, np.array(used_ids)[missing])
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, output_code)), data)

    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(output_code)), "w") as f:
        json.dump(error_set, f, indent=4)

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(output_code)), used_ids)


def main(output_dir, output_code, input_file, include_failed_cells,
        run_parallel, data_source, chirp_stimulus_codes, **kwargs):
    with open(input_file, "r") as f:
        ids = [int(line.strip("\n")) for line in f]

    run_chirp_feature_vector_extraction(output_dir, output_code,
        include_failed_cells, chirp_stimulus_codes=chirp_stimulus_codes,
        specimen_ids=ids, run_parallel=run_parallel, data_source=data_source)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectChirpFeatureVectorParameters)
    main(**module.args)