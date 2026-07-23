import logging
import os
import json
import traceback

import numpy as np
import pandas as pd
import h5py

import ipfx.lims_queries as lq
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.time_series_utils as tsu
import ipfx.error as er
import ipfx.qc_feature_extractor as qc_fex
import ipfx.qc_feature_evaluator as qc_feval
from ipfx.stimulus import StimulusType
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


def categorize_iclamp_sweeps(data_set, stimuli_names, sweep_qc_record,
        sweep_qc_option="none", specimen_id=None):

    my_sweep_qc_record = sweep_qc_record.loc[sweep_qc_record["specimen_id"] == specimen_id]
    iclamp_st = data_set.filtered_sweep_table(
        clamp_mode=data_set.CURRENT_CLAMP, stimuli=stimuli_names)

    if iclamp_st.shape[0] == 0:
        return np.array([])

    sweep_num_list = iclamp_st["sweep_number"].sort_values().unique().tolist()
    if sweep_qc_option == "none":
        return np.array(sweep_num_list)
    elif sweep_qc_option in ("passed-only", "passed-except-delta-vm", "passed-except-delta-vm-and-rms"):
        # check that sweeps exist in sweep QC record
        not_checked_list = []
        for swp_num in sweep_num_list:
            if swp_num not in my_sweep_qc_record["sweep_number"].unique():
                not_checked_list.append(swp_num)
        if len(not_checked_list) > 0:
            sweep_num_list = [sn for sn in sweep_num_list if sn not in not_checked_list]
            logging.warning("Could not find {:d} sweeps from specimen {:d} in QC record".format(len(not_checked_list), specimen_id))
        # note: choosing not to include unchecked sweeps in returned list

        # Get passed sweeps
        passed_record = my_sweep_qc_record.loc[
            my_sweep_qc_record["sweep_number"].isin(sweep_num_list) &
            my_sweep_qc_record["workflow_state"].str.endswith("passed"), :]
        passed_sweep_nums = passed_record["sweep_number"].unique()

        if sweep_qc_option == "passed-only":
            return np.sort(passed_sweep_nums).astype(int)

        # also get sweeps that only fail due to delta Vm
        failed_sweep_list = list(set(sweep_num_list) - set(passed_sweep_nums))
        if len(failed_sweep_list) == 0:
            return np.sort(passed_sweep_nums).astype(int)

        # check if only tag is "Vm delta"
        also_passing_nums = []
        for sn in failed_sweep_list:
            non_delta_vm_tag_record = my_sweep_qc_record.loc[
                (my_sweep_qc_record["sweep_number"] == sn) &
                (~my_sweep_qc_record["tag_name"].str.startswith("Vm delta")) &
                (my_sweep_qc_record["tag_name"] != "Blowout is not available"), :] # don't fail for blowout unavailable because we are considering patch-seq sweeps
            if non_delta_vm_tag_record.shape[0] == 0:
                also_passing_nums.append(sn)

        if sweep_qc_option == "passed-except-delta-vm":
            return np.sort(np.hstack([
                passed_sweep_nums,
                np.array(also_passing_nums),
                ])).astype(int)

        # Don't use LIMS-calculated RMS fail/pass - recalculate here

        # otherwise, check for having an error tag that isn't 'Vm delta'
        # or one of the RMS tags and exclude those sweeps
        rms_check_sweep_nums = []
        for sn in set(failed_sweep_list) - set(also_passing_nums):
            non_delta_vm_or_rms_tag_record = my_sweep_qc_record.loc[
                (my_sweep_qc_record["sweep_number"] == sn) &
                (~my_sweep_qc_record["tag_name"].str.startswith("Vm delta")) &
                (~my_sweep_qc_record["tag_name"].str.startswith("slow noise")) &
                (~my_sweep_qc_record["tag_name"].str.startswith("pre-noise")) &
                (~my_sweep_qc_record["tag_name"].str.startswith("post-noise")) &
                (my_sweep_qc_record["tag_name"] != "Blowout is not available"), # don't fail for blowout unavailable because we are considering patch-seq sweeps
                :]
            if non_delta_vm_tag_record.shape[0] == 0:
                rms_check_sweep_nums.append(sn)

        if len(rms_check_sweep_nums) == 0:
            # if no sweeps need to be checked, skip the rest
            return np.sort(np.hstack([
                passed_sweep_nums,
                np.array(also_passing_nums),
                ])).astype(int)

        # Now re-check each sweep's RMS
        qc_criteria = qc_feval.load_default_qc_criteria()

        # Read the lab notebook for the RMS criteria used for the sweep
        lnr = data_set._data.notebook
        numeric_fields = [c.decode('utf-8') for c in lnr.colname_number[0]]
        short_rms_fields = [f for f in numeric_fields if "S-RMS Threshold" in f]
        long_rms_fields = [f for f in numeric_fields if "L-RMS Threshold" in f]

        pass_rms_nums = []
        for sn in rms_check_sweep_nums:
            is_ramp = "Ramp" == iclamp_st.at[sn, "stimulus_name"]

            # Short RMS criterion
            s_rms_threshold = None
            for f in short_rms_fields:
                if lnr.get_value(f, sn, None) is not None:
                    s_rms_threshold = lnr.get_value(f, sn, None) * 1e3 # from V to mV
                    break
            if s_rms_threshold is None:
                s_rms_threshold = qc_criteria["pre_noise_rms_mv_max"]

            # Long RMS criterion
            l_rms_threshold = None
            for f in long_rms_fields:
                if lnr.get_value(f, sn, None) is not None:
                    l_rms_threshold = lnr.get_value(f, sn, None) * 1e3 # from V to mV
                    break
            if l_rms_threshold is None:
                l_rms_threshold = qc_criteria["slow_noise_rms_mv_max"]

            qc_features = qc_fex.current_clamp_sweep_qc_features(
                data_set.sweep(sn),
                is_ramp
            )

            if is_ramp:
                if ((qc_features["pre_noise_rms_mv"] < s_rms_threshold) &
                    (qc_features["slow_noise_rms_mv"] < l_rms_threshold)):
                    pass_rms_nums.append(sn)
            else:
                if ((qc_features["pre_noise_rms_mv"] < s_rms_threshold) &
                    (qc_features["post_noise_rms_mv"] < s_rms_threshold) &
                    (qc_features["slow_noise_rms_mv"] < l_rms_threshold)):
                    pass_rms_nums.append(sn)

        if sweep_qc_option == "passed-except-delta-vm-and-rms":
            return np.sort(np.hstack([
                passed_sweep_nums,
                np.array(also_passing_nums),
                np.array(pass_rms_nums),
                ])).astype(int)

    else:
        raise ValueError("Invalid sweep-level QC option {}".format(sweep_qc_option))


def validate_sweeps(data_set, sweep_numbers, extra_dur=0.2):
    check_sweeps = data_set.sweep_set(sweep_numbers)
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

    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
        lsq_sweeps,
        start=lsq_start,
        end=lsq_end,
        min_peak=-25,
        **dsf.detection_parameters(StimulusType.LONG_SQUARE)
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

    ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                                  est_window = [ssq_start, ssq_start + 0.001],
                                                  start=ssq_start,
                                                  end=ssq_end + spike_window,
                                                  reject_at_stim_start_interval=0.0002,
                                                  **dsf.detection_parameters(StimulusType.SHORT_SQUARE))
    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
    ssq_features = ssq_an.analyze(ssq_sweeps)

    return ssq_sweeps, ssq_features, ssq_an


def preprocess_ramp_sweeps(data_set, sweep_numbers):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No ramp sweeps available for feature extraction")

    ramp_sweeps = data_set.sweep_set(sweep_numbers)

    ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
    ramp_spx, ramp_spfx = dsf.extractors_for_sweeps(ramp_sweeps,
                                                start = ramp_start,
                                                **dsf.detection_parameters(StimulusType.RAMP))
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
