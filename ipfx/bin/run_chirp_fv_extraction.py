import argschema as ags
import ipfx.chirp as chirp
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
import ipfx.script_utils as su


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
        default=[
            "C2CHIRP180503",
            "C2CHIRP171129",
            "C2CHIRP171103",
        ],
        cli_as_single_argument=True)
    data_source = ags.fields.String(
        description="Source of NWB files ('sdk' or 'lims')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims"]
        )


def edit_ontology_data(original_ontology_data, codes_to_rename,
        new_name_tag, new_core_tag):
    ontology_data = original_ontology_data.copy()
    mask = []
    for od in ontology_data:
        mask_val = True
        for tagset in od:
            for c in codes_to_rename:
                if c in tagset and "code" in tagset:
                    mask_val = False
                    break
        mask.append(mask_val)
    ontology_data = [od for od, m in zip(ontology_data, mask) if m is True]
    ontology_data.append([
        ["code"] + codes_to_rename,
        [
          "name",
          new_name_tag,
        ],
        [
          "core",
          new_core_tag
        ]
    ])
    return ontology_data


def data_for_specimen_id(specimen_id, data_source, ontology):
    logging.debug("specimen_id: {}".format(specimen_id))

    data_set = su.dataset_for_specimen_id(specimen_id, data_source, ontology)
    if type(data_set) is dict and "error" in data_set:
        logging.warning("Problem getting AibsDataSet for specimen {:d} from LIMS".format(specimen_id))
        return data_set

    # Identify chirp sweeps
    try:
        iclamp_st = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP)
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
        result = chirp.extract_chirp_feature_vector(data_set, chirp_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when processing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=1)}}

    return result


def run_chirp_feature_vector_extraction(output_dir, output_code, include_failed_cells,
        specimen_ids, chirp_stimulus_codes, data_source="lims", run_parallel=True):
    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))

    # Include and name chirp stimulus codes in ontology
    ontology_data = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)
    edited_ontology_data = edit_ontology_data(
        ontology_data, chirp_stimulus_codes,
        new_name_tag="Chirp", new_core_tag="Core 2")
    ontology = StimulusOntology(edited_ontology_data)

    get_data_partial = partial(data_for_specimen_id,
                               data_source=data_source,
                               ontology=ontology)

    if run_parallel:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    used_ids, results, error_set = su.filter_results(specimen_ids, results)

    logging.info("Finished with {:d} processed specimens".format(len(used_ids)))

    results_dict = su.organize_results(used_ids, results)

    su.save_results_to_npy(used_ids, results_dict, output_dir, output_code)
    su.save_errors_to_json(error_set, output_dir, output_code)

    logging.info("Finished saving")


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