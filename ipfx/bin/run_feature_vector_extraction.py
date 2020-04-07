import numpy as np
import argschema as ags
import logging
import traceback
from multiprocessing import Pool
from functools import partial
import os
import h5py
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
        description="Source of NWB files ('sdk' or 'lims' or 'filesystem')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims", "filesystem"]
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

def data_for_specimen_id(
    specimen_id,
    sweep_qc_option,
    data_source,
    ontology,
    ap_window_length=0.005,
    target_sampling_rate=50000,
    file_list=None,
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
    logging.debug("specimen_id: {}".format(specimen_id))

    # Find or retrieve NWB file and ancillary info and construct an AibsDataSet object
    data_set = su.dataset_for_specimen_id(specimen_id, data_source, ontology, file_list)
    if type(data_set) is dict and "error" in data_set:
        logging.warning("Problem getting AibsDataSet for specimen {:d} from LIMS".format(specimen_id))
        return data_set

    # Identify and preprocess long square sweeps
    try:
        lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            ontology.long_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
        (lsq_sweeps,
        lsq_features,
        _,
        lsq_start,
        lsq_end) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)

    except Exception as detail:
        logging.warning("Exception when preprocessing long square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}}

    # Identify and preprocess short square sweeps
    try:
        ssq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            ontology.short_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id)
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
        ramp_ap_v, ramp_ap_dv = fv.first_ap_vectors(spiking_ramp_sweep_list,
            spiking_ramp_info_list,
            target_sampling_rate=target_sampling_rate,
            window_length=ap_window_length,
            skip_clipped=True)

        # Combine so that differences can be assessed by analyses like sPCA
        result["first_ap_v"] = np.hstack([ssq_ap_v, lsq_ap_v, ramp_ap_v])
        result["first_ap_dv"] = np.hstack([ssq_ap_dv, lsq_ap_dv, ramp_ap_dv])

        target_amplitudes = np.arange(0, 120, 20)
        supra_info_list = fv.identify_suprathreshold_spike_info(
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
    ids=None,
    file_list=None,
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

    if output_file_type == "h5":
        # Check that we can access the specified file before processing everything
        h5_file = h5py.File(os.path.join(output_dir, "fv_{}.h5".format(output_code)))
        h5_file.close()

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    get_data_partial = partial(data_for_specimen_id,
                               sweep_qc_option=sweep_qc_option,
                               data_source=data_source,
                               ontology=ontology,
                               ap_window_length=ap_window_length,
                               file_list=file_list)

    if run_parallel:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    used_ids, results, error_set = su.filter_results(specimen_ids, results)

    logging.info("Finished with {:d} processed specimens".format(len(used_ids)))

    results_dict = su.organize_results(used_ids, results)

    if output_file_type == "h5":
        su.save_results_to_h5(used_ids, results_dict, output_dir, output_code)
    elif output_file_type == "npy":
        su.save_results_to_npy(used_ids, results_dict, output_dir, output_code)
    else:
        raise ValueError("Unknown output_file_type option {} (allowed values are h5 and npy)".format(output_file_type))

    su.save_errors_to_json(error_set, output_dir, output_code)

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
