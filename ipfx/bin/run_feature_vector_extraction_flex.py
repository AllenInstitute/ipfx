import numpy as np
import pandas as pd
import argschema as ags
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import os
import h5py
import json
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
import ipfx.feature_vectors as fv
import ipfx.lims_queries as lq
import ipfx.script_utils as su
from ipfx.dataset.create import create_ephys_data_set


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
        description="Source of NWB files ('sdk', 'lims', or 'lims-nwb2')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims", "lims-nwb2", "filesystem"]
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
            "'lims-passed-except-delta-vm-and-rms': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm,"
            "but also re-calculate RMS values with current code"
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
    needed_amplitudes = ags.fields.List(
        ags.fields.Integer,
        allow_none=True,
        default=None,
        cli_as_single_argument=True
    )
    amp_tolerance = ags.fields.Float(
        default=4.
    )
    manual_fail_sweep_file = ags.fields.InputFile()



def data_for_specimen_id(
    specimen_id,
    sweep_qc_option,
    data_source,
    ontology,
    ap_window_length=0.005,
    target_sampling_rate=50000,
    needed_amplitudes=None,
    amp_tolerance=0.,
    file_list=None,
    manual_fail_sweeps=None,
):
    """
    Extract feature vector from given cell identified by the specimen_id
    Parameters
    ----------
    specimen_id : int
        cell identified
    sweep_qc_option : str
        see CollectFeatureVectorParameters input schema for details
    data_source: str
        see CollectFeatureVectorParameters input schema for details
    ontology : stimulus.StimulusOntology
        mapping of stimuli names to stimulus codes
    ap_window_length : float
        see CollectFeatureVectorParameters input schema for details
    target_sampling_rate : float
        sampling rate
    file_list : list of str
        nwbfile names
    Returns
    -------
    dict :
        features for a given cell specimen_id

    """
    logging.info(f"Starting to process {specimen_id}")
    logging.debug("specimen_id: {}".format(specimen_id))

    # Find or retrieve NWB file and ancillary info and construct an AibsDataSet object
    data_set = su.dataset_for_specimen_id(specimen_id, data_source, ontology, file_list)
    if type(data_set) is dict and "error" in data_set:
        logging.warning("Problem getting data set for specimen {:d} from LIMS".format(specimen_id))
        return data_set

    # Identify and preprocess long square sweeps
    try:
        lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            ontology.long_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)

        if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
            lsq_sweep_numbers = np.array([sn for sn in lsq_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])

        (lsq_sweeps,
        lsq_features,
        _,
        lsq_start,
        lsq_end) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)

        lsq_start_dict = None
        if type(lsq_start) is list:
            logging.info(f"Specimen {specimen_id} has different start times in long squares")
            lsq_start_dict = {lsq_sweeps.sweeps[i].sweep_number: lsq_start[i] for i in range(len(lsq_start))}
        lsq_end_dict = None
        if type(lsq_end) is list:
            logging.info(f"Specimen {specimen_id} has different end times in long squares")
            lsq_end_dict = {lsq_sweeps.sweeps[i].sweep_number: lsq_end[i] for i in range(len(lsq_end))}

    except Exception as detail:
        logging.warning("Exception when preprocessing long square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Identify and preprocess short square sweeps
    try:
        ssq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            ontology.short_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
            ssq_sweep_numbers = np.array([sn for sn in ssq_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])

        ssq_sweeps, ssq_features, _ = su.preprocess_short_square_sweeps(data_set,
            ssq_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing short square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Identify and preprocess ramp sweeps
    try:
        ramp_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            ontology.ramp_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
            ramp_sweep_numbers = np.array([sn for sn in ramp_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])
        ramp_sweeps, ramp_features, _ = su.preprocess_ramp_sweeps(data_set,
            ramp_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing ramp sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Calculate desired feature vectors
    result = {}

    if data_source == "filesystem":
        result["id"] = [specimen_id]

    try:
        if lsq_start_dict is not None:
            lsq_start = lsq_start_dict
        if lsq_end_dict is not None:
            lsq_end = lsq_end_dict

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
            lsq_sweeps, lsq_features, lsq_start, lsq_end)
        result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_end)

        if result["isi_shape"] is None:
            # Failed to calculate a shape for the first value; try other sweeps
            exclude_sweeps_for_isi = []
            while result["isi_shape"] is None:
                exclude_sweeps_for_isi.append(isi_sweep.sweep_number)
                isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
                    lsq_sweeps, lsq_features, lsq_start, lsq_end, exclude_sweep_numbers=exclude_sweeps_for_isi)
                result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_end)

        # Calculate waveforms from each type of sweep - if multiple sweeps, use the earliest
        spiking_ssq_sweep_list = [ssq_sweeps.sweeps[swp_ind]
            for swp_ind in ssq_features["common_amp_sweeps"].index]
        spiking_ssq_info_list = [ssq_features["spikes_set"][swp_ind]
            for swp_ind in ssq_features["common_amp_sweeps"].index]
        ssq_ap_v, ssq_ap_dv = fv.first_ap_vectors(spiking_ssq_sweep_list[:1],
            spiking_ssq_info_list[:1],
            target_sampling_rate=target_sampling_rate,
            window_length=ap_window_length,
            skip_clipped=True)

        rheo_ind = lsq_features["rheobase_sweep"].name
        sweep = lsq_sweeps.sweeps[rheo_ind]
        lsq_ap_v, lsq_ap_dv = fv.first_ap_vectors([sweep],
            [lsq_features["spikes_set"][rheo_ind]],
            target_sampling_rate=target_sampling_rate,
            window_length=ap_window_length)

        spiking_ramp_sweep_list = [ramp_sweeps.sweeps[swp_ind]
            for swp_ind in ramp_features["spiking_sweeps"].index]
        spiking_ramp_info_list = [ramp_features["spikes_set"][swp_ind]
            for swp_ind in ramp_features["spiking_sweeps"].index]
        ramp_ap_v, ramp_ap_dv = fv.first_ap_vectors(spiking_ramp_sweep_list[:1],
            spiking_ramp_info_list[:1],
            target_sampling_rate=target_sampling_rate,
            window_length=ap_window_length,
            skip_clipped=True)

        # Combine so that differences can be assessed by analyses like sPCA
        result["first_ap_v"] = np.hstack([ssq_ap_v, lsq_ap_v, ramp_ap_v])
        result["first_ap_dv"] = np.hstack([ssq_ap_dv, lsq_ap_dv, ramp_ap_dv])

        target_amplitudes = np.arange(0, 100, 10)
        supra_info_list, supra_sweep_numbers = fv.identify_suprathreshold_spike_info(
            lsq_features,
            target_amplitudes,
            sweep_numbers=[swp.sweep_number for swp in lsq_sweeps.sweeps],
            shift=None,
            amp_tolerance=amp_tolerance,
            needed_amplitudes=needed_amplitudes
        )

        if type(lsq_start) is dict:
            supra_lsq_start = [lsq_start[sn] if sn is not None else None for sn in supra_sweep_numbers]
        else:
            supra_lsq_start = lsq_start
        if type(lsq_end) is dict:
            supra_lsq_end = [lsq_end[sn] if sn is not None else None for sn in supra_sweep_numbers]
        else:
            supra_lsq_end = lsq_end

        actual_amps = [int(a) for a, si in zip(target_amplitudes, supra_info_list) if si is not None]
        actual_rheobase_i = int(lsq_features["rheobase_i"])

        result["long_squares_data_info"] = {"rheobase_i": actual_rheobase_i, "amplitudes_with_data": actual_amps}

        result["psth"] = fv.psth_vector(supra_info_list, supra_lsq_start, supra_lsq_end)
        result["inst_freq"] = fv.inst_freq_vector(supra_info_list, supra_lsq_start, supra_lsq_end)

        spike_feature_list = [
            "upstroke_downstroke_ratio",
            "peak_v",
            "fast_trough_v",
            "threshold_v",
            "width",
        ]
        for feature in spike_feature_list:
            result["spiking_" + feature] = fv.spike_feature_vector(feature,
                supra_info_list, supra_lsq_start, supra_lsq_end)
    except Exception as detail:
        logging.warning("Exception when processing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=None)}}

    logging.info(f"Successfully processed {specimen_id}")

    # Flush the LRU cache for the data_set object
    if hasattr(data_set, "_data") and hasattr(data_set._data, "_get_series"):
        data_set._data._get_series.cache_clear()

    return result


def run_feature_vector_extraction(
    output_dir,
    data_source,
    output_code,
    project,
    output_file_type,
    sweep_qc_option,
    include_failed_cells,
    run_parallel,
    ap_window_length,
    needed_amplitudes,
    amp_tolerance,
    ids=None,
    file_list=None,
    manual_fail_sweep_file=None,
    **kwargs
):
    """
    Extract feature vector from a list of cells and save result to the output file(s)

    Parameters
    ----------
    output_dir : str
        see CollectFeatureVectorParameters input schema for details
    data_source : str
        see CollectFeatureVectorParameters input schema for details
    output_code: str
        see CollectFeatureVectorParameters input schema for details
    project : str
        see CollectFeatureVectorParameters input schema for details
    output_file_type : str
        see CollectFeatureVectorParameters input schema for details
    sweep_qc_option: str
        see CollectFeatureVectorParameters input schema for details
    include_failed_cells: bool
        see CollectFeatureVectorParameters input schema for details
    run_parallel: bool
        see CollectFeatureVectorParameters input schema for details
    ap_window_length: float
        see CollectFeatureVectorParameters input schema for details
    ids: int
        ids associated to each cell.
    file_list: list of str
        nwbfile names
    kwargs

    Returns
    -------

    """
    if ids is not None:
        specimen_ids = ids
    elif data_source == "lims":
        specimen_ids = lq.project_specimen_ids(project, passed_only=not include_failed_cells)
    else:
        logging.error("Must specify input file if data source is not LIMS")

    if manual_fail_sweep_file is not None:
        manual_fail_df = pd.read_csv(manual_fail_sweep_file)
        manual_fail_sweep_dict = {}
        for specimen_id in manual_fail_df.specimen_id.unique():
            sweeps_for_specimen = manual_fail_df.loc[manual_fail_df.specimen_id == specimen_id, "sweep_number"].tolist()
            manual_fail_sweep_dict[specimen_id] = sweeps_for_specimen
    else:
        manual_fail_sweep_dict = None

    if output_file_type == "h5":
        # Check that we can access the specified file before processing everything
        h5_file = h5py.File(os.path.join(output_dir, "fv_{}.h5".format(output_code)), "a")
        h5_file.close()


    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    get_data_partial = partial(data_for_specimen_id,
                               sweep_qc_option=sweep_qc_option,
                               data_source=data_source,
                               ontology=ontology,
                               ap_window_length=ap_window_length,
                               needed_amplitudes=needed_amplitudes,
                               amp_tolerance=amp_tolerance,
                               file_list=file_list,
                               manual_fail_sweeps=manual_fail_sweep_dict)

    if run_parallel:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    used_ids, results, error_set = su.filter_results(specimen_ids, results)

    logging.info("Finished with {:d} processed specimens".format(len(used_ids)))

    results_dict = su.organize_results(used_ids, results, skip_keys=["long_squares_data_info"])

    if output_file_type == "h5":
        su.save_results_to_h5(used_ids, results_dict, output_dir, output_code)
    elif output_file_type == "npy":
        su.save_results_to_npy(used_ids, results_dict, output_dir, output_code)
    else:
        raise ValueError("Unknown output_file_type option {} (allowed values are h5 and npy)".format(output_file_type))

    su.save_errors_to_json(error_set, output_dir, output_code)

    amp_file_name = os.path.join(output_dir, "fv_amplitudes_with_data_{}.json".format(output_code))

    amp_info = {spec_id: r["long_squares_data_info"] for spec_id, r in zip(used_ids, results)}

    with open(amp_file_name, "w") as f:
        json.dump(amp_info, f)

    logging.info("Finished saving")


def main():
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        run_feature_vector_extraction(ids=ids, **module.args)
    else:
        run_feature_vector_extraction(**module.args)


if __name__ == "__main__": main()
