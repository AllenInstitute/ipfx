import numpy as np
import ipfx.feature_vectors as fv
import argschema as ags
import lims_utils
from ipfx.aibs_data_set import AibsDataSet
import warnings
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import os
import json


class CollectFeatureVectorParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(default="/allen/programs/celltypes/workgroups/ivscc/nathang/fv_output/")
    input = ags.fields.InputFile(default=None, allow_none=True)
    project = ags.fields.String(default="T301")
    include_failed_sweeps = ags.fields.Boolean(default=False)
    include_failed_cells = ags.fields.Boolean(default=False)
    parallel_flag = ags.fields.Boolean(default=True)
    ap_window_length = ags.fields.Float(description="Duration after threshold for AP shape (s)", default=0.003)

def project_specimen_ids(project, passed_only=True):

    SQL = """
        SELECT sp.id FROM specimens sp
        JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
        JOIN projects prj ON prj.id = sp.project_id
        WHERE prj.code = %s
        """

    if passed_only:
        SQL += " AND err.workflow_state = 'manual_passed'"

    results = lims_utils.query(SQL, (project,))

    return zip(*results)[0]


def categorize_iclamp_sweeps(data_set, stimuli_names, passed_only=False):
    # TODO - deal with pass/fail status

    iclamp_st = data_set.filtered_sweep_table(current_clamp_only=True, stimuli=stimuli_names)
    return iclamp_st["sweep_number"].sort_values().values


def data_for_specimen_id(specimen_id, passed_only, ap_window_length=0.005):
    name, roi_id, specimen_id = lims_utils.get_specimen_info_from_lims_by_id(specimen_id)
    nwb_path = lims_utils.get_nwb_path_from_lims(roi_id)
    if len(nwb_path) == 0: # could not find an NWB file
        logging.debug("No NWB file for {:d}".format(specimen_id))
        return {"error": {"type": "no_nwb", "details": ""}}

    try:
        data_set = AibsDataSet(nwb_file=nwb_path)
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


def main(ids=None, project="T301", include_failed_sweeps=True, include_failed_cells=False,
         output_dir="", parallel_flag=True, ap_window_length=0.003, **kwargs):
    if ids is not None:
        specimen_ids = ids
    else:
        specimen_ids = project_specimen_ids(project, passed_only=not include_failed_cells)

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))

    get_data_partial = partial(data_for_specimen_id,
                               passed_only=not include_failed_sweeps,
                               ap_window_length=ap_window_length)
    if parallel_flag:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    filtered_set = [(i, r) for i, r in zip(specimen_ids, results) if not "error" in r.keys()]
    error_set = [{"id": i, "error": d} for i, d in zip(specimen_ids, results) if "error" in d.keys()]
    if len(filtered_set) == 0:
        logging.info("No specimens had results")
        return

    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(project)), "w") as f:
        json.dump(error_set, f, indent=4)

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

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(project)), used_ids)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        main(ids=ids, **module.args)
    else:
        main(**module.args)