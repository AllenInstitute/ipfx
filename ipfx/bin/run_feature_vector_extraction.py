from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import pandas as pd
import argschema as ags
import warnings
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import os
import json
import h5py

import ipfx.feature_vectors as fv
import ipfx.bin.lims_queries as lq
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.time_series_utils as tsu
import ipfx.error as er

from ipfx.aibs_data_set import AibsDataSet

from allensdk.core.cell_types_cache import CellTypesCache


class CollectFeatureVectorParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(
        description="Destination directory for output files",
        default=None
    )
    input = ags.fields.InputFile(
        description=("Input file of specimen IDs (one per line)"
            "- optional if LIMS is source"),
        default=None,
        allow_none=True
    )
    data_source = ags.fields.String(
        description="Source of NWB files ('sdk' or 'lims')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims"]
        )
    output_code = ags.fields.String(
        description="Code used for naming of output files",
        default="test"
    )
    output_file_type = ags.fields.String(
        description=("File type for output - 'h5' - single HDF5 file (default) "
            "'npy' - multiple .npy files"),
        default="h5",
        validate=lambda x: x in ["h5", "npy"]
    )
    project = ags.fields.String(
        description="Project code used for LIMS query",
        default=None,
        allow_none=True
    )
    sweep_qc_option = ags.fields.String(
        description=("Sweep-level QC option - "
            "'none': use all sweeps; "
            "'lims-passed-only': check passed status with LIMS and "
            "only used passed sweeps "
            "'lims-passed-except-delta-vm': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm"
            ),
        default='none'
    )
    include_failed_cells = ags.fields.Boolean(
        description="boolean - include cells with cell-level QC failure (LIMS only)",
        default=False
    )
    run_parallel = ags.fields.Boolean(
        description="boolean - use multiprocessing",
        default=True
    )
    ap_window_length = ags.fields.Float(
        description="Duration after threshold for AP shape (s)",
        default=0.003
    )


def categorize_iclamp_sweeps(data_set, stimuli_names, sweep_qc_option=None, specimen_id=None):
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


def preprocess_long_square_sweeps(data_set, sweep_numbers, extra_dur=0.2, subthresh_min_amp=-100.):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No long_square sweeps available for feature extraction")

    check_lsq_sweeps = data_set.sweep_set(sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(check_lsq_sweeps.sweeps[0].i, check_lsq_sweeps.sweeps[0].t)
    lsq_end = lsq_start + lsq_dur

    # Check that all sweeps are long enough and not ended early
    good_sweep_numbers = [n for n, s in zip(sweep_numbers, check_lsq_sweeps.sweeps)
                              if s.t[-1] >= lsq_start + lsq_dur + extra_dur and not np.all(s.v[tsu.find_time_index(s.t, lsq_start + lsq_dur)-100:tsu.find_time_index(s.t, lsq_start + lsq_dur)] == 0)]
    lsq_sweeps = data_set.sweep_set(good_sweep_numbers)

    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
        lsq_sweeps,
        start = lsq_start,
        end = lsq_end,
        **dsf.detection_parameters(data_set.LONG_SQUARE)
    )
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
        subthresh_min_amp=subthresh_min_amp)
    lsq_features = lsq_an.analyze(lsq_sweeps)

    return lsq_sweeps, lsq_features, lsq_start, lsq_end, lsq_spx


def preprocess_short_square_sweeps(data_set, sweep_numbers):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No short square sweeps available for feature extraction")

    ssq_sweeps = data_set.sweep_set(sweep_numbers)

    ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(ssq_sweeps.sweeps[0].i, ssq_sweeps.sweeps[0].t)
    ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                                  est_window = [ssq_start, ssq_start + 0.001],
                                                  **dsf.detection_parameters(data_set.SHORT_SQUARE))
    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
    ssq_features = ssq_an.analyze(ssq_sweeps)

    return ssq_sweeps, ssq_features


def preprocess_ramp_sweeps(data_set, sweep_numbers):
    if len(sweep_numbers) == 0:
        raise er.FeatureError("No ramp sweeps available for feature extraction")

    ramp_sweeps = data_set.sweep_set(sweep_numbers)

    ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
    ramp_spx, ramp_spfx = dsf.extractors_for_sweeps(ramp_sweeps,
                                                start = ramp_start,
                                                **dsf.detection_parameters(data_set.RAMP))
    ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
    ramp_features = ramp_an.analyze(ramp_sweeps)

    return ramp_sweeps, ramp_features


def data_for_specimen_id(specimen_id, sweep_qc_option, data_source,
        ap_window_length=0.005, target_sampling_rate=50000):
    logging.debug("specimen_id: {}".format(specimen_id))

    # Find or retrieve NWB file and ancillary info and construct an AibsDataSet object
    if data_source == "lims":
        nwb_path, h5_path = lims_nwb_information(specimen_id)
        if type(nwb_path) is dict and "error" in nwb_path:
            logging.warning("Problem getting NWB file for specimen {:d} from LIMS".format(specimen_id))
            return nwb_path

        try:
            data_set = AibsDataSet(nwb_file=nwb_path, h5_file=h5_path)
            ontology = data_set.ontology
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} from LIMS".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    elif data_source == "sdk":
        nwb_path, sweep_info = sdk_nwb_information(specimen_id)
        try:
            data_set = AibsDataSet(nwb_file=nwb_path, sweep_info=sweep_info)
            ontology = data_set.ontology
        except Exception as detail:
            logging.warning("Exception when loading specimen {:d} via Allen SDK".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}
    else:
        logging.error("invalid data source specified ({})".format(data_source))

    # Identify and preprocess long square sweeps
    try:
        lsq_sweep_numbers = categorize_iclamp_sweeps(data_set,
            ontology.long_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        (lsq_sweeps,
        lsq_features,
        lsq_start,
        lsq_end,
        lsq_spx) = preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing long square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Identify and preprocess short square sweeps
    try:
        ssq_sweep_numbers = categorize_iclamp_sweeps(data_set,
            ontology.short_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        ssq_sweeps, ssq_features = preprocess_short_square_sweeps(data_set,
            ssq_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing short square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Identify and preprocess ramp sweeps
    try:
        ramp_sweep_numbers = categorize_iclamp_sweeps(data_set,
            ontology.ramp_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        ramp_sweeps, ramp_features = preprocess_ramp_sweeps(data_set,
            ramp_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing ramp sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Calculate desired feature vectors
    result = {}
    try:
        (subthresh_hyperpol_dict,
        hyperpol_deflect_dict) = fv.identify_subthreshold_hyperpol_with_amplitudes(lsq_features,
            lsq_sweeps)
        target_amps_for_step_subthresh = [-90, -70, -50, -30, -10]
        result["step_subthresh"] = fv.step_subthreshold(
            subthresh_hyperpol_dict, target_amps_for_step_subthresh,
            lsq_start, lsq_end, amp_tolerance=5)
        result["subthresh_norm"] = fv.subthresh_norm(subthresh_hyperpol_dict, hyperpol_deflect_dict,
            lsq_start, lsq_end)
        (subthresh_depol_dict,
        depol_deflect_dict) = fv.identify_subthreshold_depol_with_amplitudes(lsq_features,
            lsq_sweeps)
        result["subthresh_depol_norm"] = fv.subthresh_depol_norm(subthresh_depol_dict,
            depol_deflect_dict, lsq_start, lsq_end)
        isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
            lsq_sweeps, lsq_features, lsq_end - lsq_start)
        result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_end)

        # Calculate waveforms from each type of sweep
        spiking_ssq_sweep_list = [ssq_sweeps.sweeps[swp_ind]
            for swp_ind in ssq_features["common_amp_sweeps"].index]
        spiking_ssq_info_list = [ssq_features["spikes_set"][swp_ind]
            for swp_ind in ssq_features["common_amp_sweeps"].index]
        ssq_ap_v, ssq_ap_dv = fv.first_ap_vectors(spiking_ssq_sweep_list,
            spiking_ssq_info_list,
            window_length=ap_window_length)

        rheo_ind = lsq_features["rheobase_sweep"].name
        sweep = lsq_sweeps.sweeps[rheo_ind]
        lsq_ap_v, lsq_ap_dv = fv.first_ap_vectors([sweep],
            [lsq_features["spikes_set"][rheo_ind]],
            window_length=ap_window_length)

        spiking_ramp_sweep_list = [ramp_sweeps.sweeps[swp_ind]
            for swp_ind in ramp_features["spiking_sweeps"].index]
        spiking_ramp_info_list = [ramp_features["spikes_set"][swp_ind]
            for swp_ind in ramp_features["spiking_sweeps"].index]
        ramp_ap_v, ramp_ap_dv = fv.first_ap_vectors(spiking_ramp_sweep_list,
            spiking_ramp_info_list,
            window_length=ap_window_length)

        # Combine so that differences can be assessed by analyses like sPCA
        result["first_ap_v"] = np.hstack([ssq_ap_v, lsq_ap_v, ramp_ap_v])
        result["first_ap_dv"] = np.hstack([ssq_ap_dv, lsq_ap_dv, ramp_ap_dv])


#         result["spiking"] = spiking_features(lsq_sweeps, lsq_features, lsq_spike_extractor,
#                                              lsq_start, lsq_end,
#                                              feature_width, rate_width)

        target_amplitudes = np.arange(0, 120, 20)
        supra_info_list = fv.identify_suprathreshold_sweep_sequence(
            lsq_features, target_amplitudes, shift=10)
        result["psth"] = fv.psth_vector(supra_info_list, lsq_start, lsq_end)
        result["inst_freq"] = fv.inst_freq_vector(supra_info_list, lsq_start, lsq_end)

        spike_feature_list = [
            "upstroke_downstroke_ratio",
            "peak_v",
            "fast_trough_v",
            "threshold_v",
            "width",
        ]
        for feature in spike_feature_list:
            result["spiking_" + feature] = fv.spike_feature_vector(feature,
                supra_info_list, lsq_start, lsq_end)
    except Exception as detail:
        logging.warning("Exception when processing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=None)}}

    return result


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


def save_to_npy(specimen_ids, results_dict, output_dir, output_code):
    k_sizes = {}
    for k in results_dict:
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, output_code)), results_dict[k])
    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(output_code)), specimen_ids)


def save_to_h5(specimen_ids, results_dict, output_dir, output_code):
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


def run_feature_vector_extraction(output_dir, data_source, output_code, project,
        output_file_type, sweep_qc_option, include_failed_cells, run_parallel,
        ap_window_length, ids=None, **kwargs):
    if ids is not None:
        specimen_ids = ids
    elif data_source == "lims":
        specimen_ids = lq.project_specimen_ids(project, passed_only=not include_failed_cells)
    else:
        logging.error("Must specify input file if data source is not LIMS")

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    get_data_partial = partial(data_for_specimen_id,
                               sweep_qc_option=sweep_qc_option,
                               data_source=data_source,
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

    results_dict = organize_results(used_ids, results)

    if output_file_type == "h5":
        save_to_h5(used_ids, results_dict, output_dir, output_code)
    elif output_file_type == "npy":
        save_to_npy(used_ids, results_dict, output_dir, output_code)
    else:
        raise ValueError("Unknown output_file_type option {} (allowed values are h5 and npy)".format(output_file_type))

    with open(os.path.join(output_dir, "fv_errors_{:s}.json".format(output_code)), "w") as f:
        json.dump(error_set, f, indent=4)


def main():
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        run_feature_vector_extraction(ids=ids, **module.args)
    else:
        run_feature_vector_extraction(**module.args)


if __name__ == "__main__": main()
