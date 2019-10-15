import numpy as np
from ipfx.ephys_data_set import SweepSet
import ipfx.feature_vectors as fv
from ipfx.synphys import *

import argschema as ags
import os
import json
import logging
import traceback
from multiprocessing import Pool


class SynPhysFeatureVectorSchema(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(default="output")
    input = ags.fields.InputFile(default=None, allow_none=True)
    project = ags.fields.String(default="")
    # sweep_qc_option = ags.fields.String(default=None, allow_none=True)
    run_parallel = ags.fields.Boolean(default=True)
    ap_window_length = ags.fields.Float(description="Duration after threshold for AP shape (s)", default=0.003)


def run_mpa_cell(mpid, ap_window_length=0.003):
    try:
        cell = cell_from_mpid(mpid)
        # data is a MiesNwb instance
        nwb = cell.experiment.data
        channel = cell.electrode.device_id
        sweeps_dict = sweeps_dict_from_cell(cell)
        supra_sweep_ids = sweeps_dict['If_Curve_DA_0']
        sub_sweep_ids = sweeps_dict['TargetV_DA_0']
        lsq_supra_sweep_list = [MPSweep(nwb.contents[i][channel]) for i in supra_sweep_ids]
        lsq_sub_sweep_list = [MPSweep(nwb.contents[i][channel]) for i in sub_sweep_ids]
        # filter out sweeps that don't have stim epoch
        # (probably zero amplitude)
        lsq_supra_sweep_list = [sweep for sweep in lsq_supra_sweep_list 
            if 'stim' in sweep.epochs]
        lsq_sub_sweep_list = [sweep for sweep in lsq_sub_sweep_list 
            if 'stim' in sweep.epochs]

        lsq_supra_sweeps = SweepSet(lsq_supra_sweep_list)
        lsq_sub_sweeps = SweepSet(lsq_sub_sweep_list)
        all_sweeps = [lsq_supra_sweeps, lsq_sub_sweeps]
        for sweepset in all_sweeps:
            sweepset.align_to_start_of_epoch('stim')
        
        # We may not need this - do durations actually vary for a given cell?
        lsq_supra_dur = min_duration_of_sweeplist(lsq_supra_sweep_list)
        lsq_sub_dur = min_duration_of_sweeplist(lsq_sub_sweep_list)

    except Exception as detail:
        logging.warn("Exception when processing specimen {}".format(mpid))
        logging.warn(detail)
        return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}

    try:
        all_features = fv.extract_multipatch_feature_vectors(lsq_supra_sweeps, 0., lsq_supra_dur,
                                                         lsq_sub_sweeps, 0., lsq_sub_dur,
                                                         ap_window_length=ap_window_length)
    except Exception as detail:
        logging.warn("Exception when processing specimen {}".format(mpid))
        logging.warn(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=None)}}
    return all_features

def run_cells(ids=None, output_dir="", project='mp_test', run_parallel=True, 
            ap_window_length=0.003, max_count=None, **kwargs):

    if ids is not None:
        specimen_ids = ids
    else:
        specimen_ids = mp_project_cell_ids(project, max_count=max_count, filter_cells=True)
    
    if run_parallel:
        pool = Pool()
        results = pool.map(run_mpa_cell, specimen_ids)
    else:
        results = map(run_mpa_cell, specimen_ids)

    filtered_set = [(i, r) for i, r in zip(specimen_ids, results) if not "error" in r.keys()]
    error_set = [{"id": i, "error": d} for i, d in zip(specimen_ids, results) if "error" in d.keys()]
    
    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(project)), "w") as f:
        json.dump(error_set, f, indent=4)

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
        if len(data.shape) == 1: # it'll be 1D if there's just one specimen
            data = np.reshape(data, (1, -1))
        if data.shape[0] < len(used_ids):
            logging.warn("Missing data!")
            missing = np.array([k not in r for r in results])
            print k, np.array(used_ids)[missing]
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, project)), data)

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(project)), used_ids)


def main():
    module = ags.ArgSchemaParser(schema_type=SynPhysFeatureVectorSchema)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [line.strip("\n") for line in f]
        run_cells(ids=ids, **module.args)
    else:
        run_cells(**module.args)


if __name__ == "__main__": main()