import logging
import os
import json
import traceback

import numpy as np
import pandas as pd
import h5py

from allensdk.core.cell_types_cache import CellTypesCache

import ipfx.lims_queries as lq
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.time_series_utils as tsu
import ipfx.error as er
from ipfx.sweep import SweepSet
from ipfx.dataset.create import create_ephys_data_set


def lims_nwb_information(specimen_id):
    _, roi_id, _ = lq.get_specimen_info_from_lims_by_id(specimen_id)
    if roi_id is None:
        logging.warning("No ephys ROI result found for {:d}".format(specimen_id))
        return {"error": {"type": "no_ephys_roi_result", "details": "roi ID was None"}}, None

    nwb_path = lq.get_nwb_path_from_lims(roi_id)
    if (nwb_path is None) or (len(nwb_path) == 0): # could not find an NWB file
        logging.warning("No NWB file for {:d}".format(specimen_id))
        return {"error": {"type": "no_nwb", "details": "empty nwb path"}}, None

    # Check if NWB has lab notebook information, or if additional hdf5 file is needed
    h5_path = None
    try:
        with h5py.File(nwb_path, "r") as h5:
            if "general/labnotebook" not in h5:
                try:
                    h5_path = lq.get_igorh5_path_from_lims(roi_id)
                except Exception as detail:
                    logging.warning("Exception when loading h5 file for {:d}".format(specimen_id))
                    logging.warning(detail)
                    return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}, None
    except:
        logging.warning("Could not open NWB file for {:d}".format(specimen_id))
        return {"error": {"type": "no_nwb", "details": ""}}, None
    return nwb_path, h5_path


def sdk_nwb_information(specimen_id):
    ctc = CellTypesCache()
    nwb_data_set = ctc.get_ephys_data(specimen_id)
    sweep_info = ctc.get_ephys_sweeps(specimen_id)
    return nwb_data_set.file_name, sweep_info


def dataset_for_specimen_id(specimen_id, data_source, ontology, file_list=None):
    if data_source == "lims":
        nwb_path, h5_path = lims_nwb_information(specimen_id)
        if type(nwb_path) is dict and "error" in nwb_path:
            logging.warning("Problem getting NWB file for specimen {:d} from LIMS".format(specimen_id))
            return nwb_path

        try:
            data_set = create_ephys_data_set(
                nwb_file=nwb_path, ontology=ontology)
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} from LIMS".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    elif data_source == "sdk":
        nwb_path, sweep_info = sdk_nwb_information(specimen_id)
        try:
            data_set = create_ephys_data_set(
                nwb_file=nwb_path, sweep_info=sweep_info, ontology=ontology)
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} via Allen SDK".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    elif data_source == "filesystem":
        nwb_path = file_list[specimen_id]
        try:
            data_set = create_ephys_data_set(nwb_file=nwb_path)
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} via file system".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    else:
        logging.error("invalid data source specified ({})".format(data_source))

    return data_set


def categorize_iclamp_sweeps(data_set, stimuli_names, sweep_qc_option="none", specimen_id=None):
    exist_sql = """
        select swp.sweep_number from ephys_sweeps swp
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
    """

    passed_sql = """
        select swp.sweep_number from ephys_sweeps swp
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
        and swp.workflow_state like '%%passed'
    """

    passed_except_delta_vm_sql = """
        select swp.sweep_number, tag.name
        from ephys_sweeps swp
        join ephys_sweep_tags_ephys_sweeps estes on estes.ephys_sweep_id = swp.id
        join ephys_sweep_tags tag on tag.id = estes.ephys_sweep_tag_id
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
    """

    iclamp_st = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP, stimuli=stimuli_names)

    if iclamp_st.shape[0] == 0:
        return np.array([])

    if sweep_qc_option == "none":
        return iclamp_st["sweep_number"].sort_values().values
    elif sweep_qc_option == "lims-passed-only":
        # check that sweeps exist in LIMS
        sweep_num_list = iclamp_st["sweep_number"].sort_values().tolist()
        results = lq.query(exist_sql, (specimen_id, sweep_num_list))
        res_nums = pd.DataFrame(results, columns=["sweep_number"])["sweep_number"].tolist()
        not_checked_list = []
        for swp_num in sweep_num_list:
            if swp_num not in res_nums:
                logging.debug("Could not find sweep {:d} from specimen {:d} in LIMS for QC check".format(swp_num, specimen_id))
                not_checked_list.append(swp_num)

        # Get passed sweeps
        results = lq.query(passed_sql, (specimen_id, sweep_num_list))
        results_df = pd.DataFrame(results, columns=["sweep_number"])
        passed_sweep_nums = results_df["sweep_number"].values
        return np.sort(np.hstack([passed_sweep_nums, np.array(not_checked_list)])) # deciding to keep non-checked sweeps for now
    elif sweep_qc_option == "lims-passed-except-delta-vm":
        # check that sweeps exist in LIMS
        sweep_num_list = iclamp_st["sweep_number"].sort_values().tolist()
        results = lq.query(exist_sql, (specimen_id, sweep_num_list))
        res_nums = pd.DataFrame(results, columns=["sweep_number"])["sweep_number"].tolist()

        not_checked_list = []
        for swp_num in sweep_num_list:
            if swp_num not in res_nums:
                logging.debug("Could not find sweep {:d} from specimen {:d} in LIMS for QC check".format(swp_num, specimen_id))
                not_checked_list.append(swp_num)

        # get straight-up passed sweeps
        results = lq.query(passed_sql, (specimen_id, sweep_num_list))
        results_df = pd.DataFrame(results, columns=["sweep_number"])
        passed_sweep_nums = results_df["sweep_number"].values

        # also get sweeps that only fail due to delta Vm
        failed_sweep_list = list(set(sweep_num_list) - set(passed_sweep_nums))
        if len(failed_sweep_list) == 0:
            return np.sort(passed_sweep_nums)
        results = lq.query(passed_except_delta_vm_sql, (specimen_id, failed_sweep_list))
        results_df = pd.DataFrame(results, columns=["sweep_number", "name"])

        # not all cells have tagged QC status - if there are no tags assume the
        # fail call is correct and exclude those sweeps
        tagged_mask = np.array([sn in results_df["sweep_number"].tolist() for sn in failed_sweep_list])

        # otherwise, check for having an error tag that isn't 'Vm delta'
        # and exclude those sweeps
        has_non_delta_tags = np.array([np.any((results_df["sweep_number"].values == sn) &
            (results_df["name"].values != "Vm delta")) for sn in failed_sweep_list])

        also_passing_nums = np.array(failed_sweep_list)[tagged_mask & ~has_non_delta_tags]

        return np.sort(np.hstack([passed_sweep_nums, also_passing_nums, np.array(not_checked_list)]))
    else:
        raise ValueError("Invalid sweep-level QC option {}".format(sweep_qc_option))


def validate_sweeps(data_set, sweep_numbers, extra_dur=0.2):
    check_sweeps = data_set.sweep_set(sweep_numbers)
    check_sweeps.select_epoch("recording")
    valid_sweep_stim = []
    start = None
    dur = None
    for swp in check_sweeps.sweeps:
        if len(swp.t) == 0:
            valid_sweep_stim.append(False)
            continue

        swp_start, swp_dur, _, _, _ = stf.get_stim_characteristics(swp.i, swp.t)
        if swp_start is None:
            valid_sweep_stim.append(False)
        else:
            start = swp_start
            dur = swp_dur
            valid_sweep_stim.append(True)
    if start is None:
        # Could not find any sweeps to define stimulus interval
        return [], None, None

    end = start + dur

    # Check that all sweeps are long enough and not ended early
    good_sweeps = [s for s, v in zip(check_sweeps.sweeps, valid_sweep_stim)
                              if s.t[-1] >= end + extra_dur
                              and v is True
                              and not np.all(s.v[tsu.find_time_index(s.t, end)-100:tsu.find_time_index(s.t, end)] == 0)]
    return SweepSet(sweeps=good_sweeps), start, end


def preprocess_long_square_sweeps(data_set, sweep_numbers, extra_dur=0.2, subthresh_min_amp=-100.):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No long square sweeps available for feature extraction")

    lsq_sweeps, lsq_start, lsq_end = validate_sweeps(data_set, sweep_numbers, extra_dur=extra_dur)
    if len(lsq_sweeps.sweeps) == 0:
        raise er.FeatureError("No long square sweeps were long enough or did not end early")
    lsq_sweeps.select_epoch("recording")

    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
        lsq_sweeps,
        start=lsq_start,
        end=lsq_end,
        min_peak=-25,
        **dsf.detection_parameters(data_set.LONG_SQUARE)
    )
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
        subthresh_min_amp=subthresh_min_amp)
    lsq_features = lsq_an.analyze(lsq_sweeps)

    return lsq_sweeps, lsq_features, lsq_an, lsq_start, lsq_end


def preprocess_short_square_sweeps(data_set, sweep_numbers, extra_dur=0.2, spike_window=0.05):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No short square sweeps available for feature extraction")

    ssq_sweeps, ssq_start, ssq_end  = validate_sweeps(data_set, sweep_numbers, extra_dur=extra_dur)
    if len(ssq_sweeps.sweeps) == 0:
        raise er.FeatureError("No short square sweeps were long enough or did not end early")
    ssq_sweeps.select_epoch("recording")

    ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                                  est_window = [ssq_start, ssq_start + 0.001],
                                                  start=ssq_start,
                                                  end=ssq_end + spike_window,
                                                  reject_at_stim_start_interval=0.0002,
                                                  **dsf.detection_parameters(data_set.SHORT_SQUARE))
    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
    ssq_features = ssq_an.analyze(ssq_sweeps)

    return ssq_sweeps, ssq_features, ssq_an


def preprocess_ramp_sweeps(data_set, sweep_numbers):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No ramp sweeps available for feature extraction")

    ramp_sweeps = data_set.sweep_set(sweep_numbers)
    ramp_sweeps.select_epoch("recording")

    ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
    ramp_spx, ramp_spfx = dsf.extractors_for_sweeps(ramp_sweeps,
                                                start = ramp_start,
                                                **dsf.detection_parameters(data_set.RAMP))
    ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
    ramp_features = ramp_an.analyze(ramp_sweeps)

    return ramp_sweeps, ramp_features, ramp_an


def filter_results(specimen_ids, results):
    filtered_set = [(i, r) for i, r in zip(specimen_ids, results) if not "error" in r.keys()]
    error_set = [{"id": i, "error": d} for i, d in zip(specimen_ids, results) if "error" in d.keys()]
    if len(filtered_set) == 0:
        logging.info("No specimens had results")
        return

    used_ids, results = zip(*filtered_set)
    return used_ids, results, error_set


def organize_results(specimen_ids, results):
    """Build dictionary of results, filling data from cells with appropriate-length
        nan arrays where needed"""
    result_sizes = {}
    output = {}
    all_keys = np.unique(np.concatenate([list(r.keys()) for r in results]))

    for k in all_keys:
        if k not in result_sizes:
            for r in results:
                if k in r and r[k] is not None:
                    result_sizes[k] = len(r[k])
        data = np.array([r[k] if k in r else np.nan * np.zeros(result_sizes[k])
                        for r in results])
        output[k] = data

    return output


def save_results_to_npy(specimen_ids, results_dict, output_dir, output_code):
    k_sizes = {}
    for k in results_dict:
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, output_code)), results_dict[k])
    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(output_code)), specimen_ids)


def save_results_to_h5(specimen_ids, results_dict, output_dir, output_code):
    ids_arr = np.array(specimen_ids)
    h5_file = h5py.File(os.path.join(output_dir, "fv_{}.h5".format(output_code)), "w")
    for k in results_dict:
        data = results_dict[k]
        dset = h5_file.create_dataset(k, data.shape, dtype=data.dtype,
            compression="gzip")
        dset[...] = data
    dset = h5_file.create_dataset("ids", ids_arr.shape,
        dtype=ids_arr.dtype, compression="gzip")
    dset[...] = ids_arr
    h5_file.close()


def save_errors_to_json(error_set, output_dir, output_code):
    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(output_code)), "w") as f:
        json.dump(error_set, f, indent=4)
