import numpy as np
import ipfx.feature_vectors as fv
import argschema as ags
import lims_queries as lq
from ipfx.aibs_data_set import AibsDataSet
import warnings
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import os
import json
import h5py


class CollectFeatureVectorParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(default=None)
    input = ags.fields.InputFile(default=None, allow_none=True)
    project = ags.fields.String(default="T301")
    include_failed_sweeps = ags.fields.Boolean(default=False)
    include_failed_cells = ags.fields.Boolean(default=False)
    run_parallel = ags.fields.Boolean(default=True)
    ap_window_length = ags.fields.Float(description="Duration after threshold for AP shape (s)", default=0.003)


def categorize_iclamp_sweeps(data_set, stimuli_names, passed_only=False):
    # TODO - deal with pass/fail status

    iclamp_st = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP, stimuli=stimuli_names)
    return iclamp_st["sweep_number"].sort_values().values


def data_for_specimen_id(specimen_id, passed_only, ap_window_length=0.005):
    name, roi_id, specimen_id = lq.get_specimen_info_from_lims_by_id(specimen_id)
    nwb_path = lq.get_nwb_path_from_lims(roi_id)
    if len(nwb_path) == 0: # could not find an NWB file
        logging.debug("No NWB file for {:d}".format(specimen_id))
        return {"error": {"type": "no_nwb", "details": ""}}

    # Check if NWB has lab notebook information, or if additional hdf5 file is needed
    h5_path = None
    with h5py.File(nwb_path, "r") as h5:
        if "general/labnotebook" not in h5:
            h5_path = lq.get_igorh5_path_from_lims(roi_id)

    try:
        data_set = AibsDataSet(nwb_file=nwb_path, h5_file=h5_path)
        ontology = data_set.ontology
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
        return {"error": {"type": "dataset", "details": traceback.format_exc(limit=1)}}

    try:
        lsq_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.long_square_names)
        ssq_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.short_square_names)
        ramp_sweep_numbers = categorize_iclamp_sweeps(data_set, ontology.ramp_names)
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=1)}}

    try:
        result = fv.extract_feature_vectors(data_set, ramp_sweep_numbers, ssq_sweep_numbers, lsq_sweep_numbers,
                                            ap_window_length=ap_window_length)
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=1)}}

    return result


def run_feature_vector_extraction(ids=None, project="T301", include_failed_sweeps=True, include_failed_cells=False,
         output_dir="", run_parallel=True, ap_window_length=0.003, **kwargs):
    if ids is not None:
        specimen_ids = ids
    else:
        specimen_ids = lq.project_specimen_ids(project, passed_only=not include_failed_cells)

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    get_data_partial = partial(data_for_specimen_id,
                               passed_only=not include_failed_sweeps,
                               ap_window_length=ap_window_length)
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
            logging.warn("Missing data!")
            missing = np.array([k not in r for r in results])
            print k, np.array(used_ids)[missing]
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, project)), data)

    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(project)), "w") as f:
        json.dump(error_set, f, indent=4)

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(project)), used_ids)


def main():
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        run_feature_vector_extraction(ids=ids, **module.args)
    else:
        run_feature_vector_extraction(**module.args)


if __name__ == "__main__": main()
