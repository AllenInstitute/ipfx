import logging
import os
import json
import traceback

import argschema as ags
import numpy as np
import pandas as pd

import ipfx.lims_queries as lq
import ipfx.json_utilities as ju
import ipfx.script_utils as su
import ipfx.feature_vectors as fv

from concurrent.futures import ProcessPoolExecutor
from functools import partial

from ipfx.dataset.create import create_ephys_data_set


class StartEndDurationSchema(ags.schemas.DefaultSchema):
    before = ags.fields.Float(
        description="duration to extend before stimulus",
        default=0.2,
    )
    after = ags.fields.Float(
        description="duration to extend after stimulus",
        default=0.2,
    )


class ExtendDurationSchema(ags.schemas.DefaultSchema):
    step_subthresh = ags.fields.Nested(StartEndDurationSchema,
        required=True,
        default={"before": 0.2, "after": 0.2},
        description="parameters for extending duration around step subthreshold analysis",
    )
    subthresh_norm = ags.fields.Nested(StartEndDurationSchema,
        required=True,
        default={"before": 0.2, "after": 0.2},
        description="parameters for extending duration around normalized subthreshold analysis",
    )


class ApWaveformSchema(ags.schemas.DefaultSchema):
    use = ags.fields.Boolean(
        default=True,
        description="whether to use AP from stimulus type",
    )
    duration = ags.fields.Float(
        default=0.003,
        description="Duration after threshold for AP shape (s)",
    )


class ApWaveformForStimuliSchema(ags.schemas.DefaultSchema):
    ssq = ags.fields.Nested(ApWaveformSchema,
        default={},
        description="analysis parameters for short square AP waveform",
    )
    lsq = ags.fields.Nested(ApWaveformSchema,
        default={},
        description="analysis parameters for long square AP waveform",
    )
    ramp = ags.fields.Nested(ApWaveformSchema,
        default={},
        description="analysis parameters for ramp AP waveform",
    )


class CollectFeatureVectorParameters(ags.ArgSchema):
    output_dir = ags.fields.OutputDir(
        description="Destination directory for output files",
        default="."
    )
    output_code = ags.fields.String(
        description="Code used for naming of output files",
        default="test"
    )
    specimen_id_file = ags.fields.InputFile(
        description=("Input file of specimen IDs (one per line)"),
    )
    nwb_path_file = ags.fields.InputFile(
        description=("JSON file with paths to each specimen's NWB file - "
            "if not supplied, LIMS will be queried for them"),
        default=None,
        allow_none=True,
    )
    sweep_qc_record_file = ags.fields.InputFile(
        description=("File with sweep QC status and tags - "
            "if not supplied, LIMS will be queried for them"),
        default=None,
        allow_none=True,
    )
    manual_fail_sweep_file = ags.fields.InputFile(
        description=("File with manual sweep failure information"),
        default=None,
        allow_none=True,
    )
    sweep_qc_option = ags.fields.String(
        description=("Sweep-level QC option - "
            "'none': use all sweeps; "
            "'passed-only': check passed status with LIMS and "
            "only used passed sweeps "
            "'passed-except-delta-vm': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm"
            "'passed-except-delta-vm-and-rms': check status with LIMS and "
            "use passed sweeps and sweeps where only failure criterion is delta_vm,"
            "but also re-calculate RMS values with current code"
            ),
        default='none'
    )
    extract_from_ramp = ags.fields.Boolean(
        description="whether to run analysis on ramp sweep",
        default=True,
    )
    amp_tolerance = ags.fields.Float(
        description="how much deviation from expected stimulus amplitudes is acceptable (in pA)",
        default=4.
    )
    additional_fvs = ags.fields.List(ags.fields.String,
        allow_none=True,
        default=[],
        cli_as_single_argument=True,
    )
    extend_durations = ags.fields.Nested(ExtendDurationSchema,
        description="parameters for extending time windows for analyses",
        default={},
    )
    ap_waveforms = ags.fields.Nested(ApWaveformForStimuliSchema,
        default={},
        description="parameters for AP waveform analysis",
    )
    needed_amplitudes = ags.fields.List(
        ags.fields.Integer,
        allow_none=True,
        default=None,
        cli_as_single_argument=True
    )
    run_parallel = ags.fields.Boolean(
        description="boolean - use multiprocessing",
        default=True
    )



def data_for_specimen_id(
    specimen_id,
    sweep_qc_option,
    sweep_qc_record,
    file_list,
    ap_waveforms,
    extend_durations,
    extract_from_ramp,
    additional_fvs,
    target_sampling_rate=50000,
    needed_amplitudes=None,
    amp_tolerance=0.,
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
    sweep_qc_record: DataFrame
        sweep status and error tag dataframe
    data_source: str
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
    logging.debug(f"Starting to process specimen id: {specimen_id}")

    try:
        data_set = create_ephys_data_set(nwb_file=file_list[specimen_id])
    except Exception as detail:
        logging.warning("Exception when creating data set for specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "data_set", "details": traceback.format_exc(limit=None)}, "specimen_id": specimen_id}

    # Identify and preprocess long square sweeps
    try:
        lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            data_set.ontology.long_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id, sweep_qc_record=sweep_qc_record)

        if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
            lsq_sweep_numbers = np.array([sn for sn in lsq_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])

        (lsq_sweeps,
        lsq_features,
        _,
        lsq_stim_timing) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)

        # Create stimulus timing dictionary keyed on sweep number
        lsq_stim_timing_dict = {lsq_sweeps.sweeps[i].sweep_number: lsq_stim_timing[i]
            for i in range(len(lsq_stim_timing))}
    except Exception as detail:
        logging.warning("Exception when preprocessing long square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}, "specimen_id": specimen_id}


    # Identify and preprocess short square sweeps
    try:
        ssq_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
            data_set.ontology.short_square_names, sweep_qc_option=sweep_qc_option,
            specimen_id=specimen_id, sweep_qc_record=sweep_qc_record)

        if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
            ssq_sweep_numbers = np.array([sn for sn in ssq_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])

        ssq_sweeps, ssq_features, _ = su.preprocess_short_square_sweeps(data_set,
            ssq_sweep_numbers)
    except Exception as detail:
        logging.warning("Exception when preprocessing short square sweeps from specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None)}, "specimen_id": specimen_id}

    # Identify and preprocess ramp sweeps
    if extract_from_ramp:
        logging.debug("Identifying and processing ramp sweeps")
        try:
            ramp_sweep_numbers = su.categorize_iclamp_sweeps(data_set,
                data_set.ontology.ramp_names, sweep_qc_option=sweep_qc_option,
                specimen_id=specimen_id, sweep_qc_record=sweep_qc_record)
            if manual_fail_sweeps is not None and specimen_id in manual_fail_sweeps:
                ramp_sweep_numbers = np.array([sn for sn in ramp_sweep_numbers if sn not in manual_fail_sweeps[specimen_id]])
            ramp_sweeps, ramp_features, _ = su.preprocess_ramp_sweeps(data_set,
                ramp_sweep_numbers)
        except Exception as detail:
            logging.warning("Exception when preprocessing ramp sweeps from specimen {:d}".format(specimen_id))
            logging.warning(detail)
            return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=None), "specimen_id": specimen_id}}

    # Calculate desired feature vectors
    result = {"id": specimen_id}

    try:
        (subthresh_hyperpol_dict,
        hyperpol_deflect_dict) = fv.identify_subthreshold_hyperpol_with_amplitudes(lsq_features,
            lsq_sweeps)
        target_amps_for_step_subthresh = [-90, -70, -50, -30, -10]
        result["step_subthresh"] = fv.step_subthreshold(
            subthresh_hyperpol_dict, target_amps_for_step_subthresh,
            lsq_stim_timing_dict, amp_tolerance=amp_tolerance,
            extend_duration_before=extend_durations["step_subthresh"]["before"],
            extend_duration_after=extend_durations["step_subthresh"]["after"],
        )
        result["subthresh_norm"] = fv.subthresh_norm(subthresh_hyperpol_dict, hyperpol_deflect_dict,
            lsq_stim_timing_dict,
            extend_duration_before=extend_durations["subthresh_norm"]["before"],
            extend_duration_after=extend_durations["subthresh_norm"]["after"],
        )
        if "subthresh_rebound" in additional_fvs:
            result["subthresh_rebound"] = fv.subthresh_rebound(
                subthresh_hyperpol_dict,
                lsq_stim_timing_dict, dur=0.3,
            )

        (subthresh_depol_dict,
        depol_deflect_dict) = fv.identify_subthreshold_depol_with_amplitudes(lsq_features,
            lsq_sweeps)
        result["subthresh_depol_norm"] = fv.subthresh_depol_norm(subthresh_depol_dict,
            depol_deflect_dict, lsq_stim_timing_dict)
        isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
            lsq_sweeps, lsq_features, lsq_stim_timing_dict)
        result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_stim_timing_dict)

        if result["isi_shape"] is None:
            # Failed to calculate a shape for the first value; try other sweeps
            exclude_sweeps_for_isi = []
            while result["isi_shape"] is None:
                exclude_sweeps_for_isi.append(isi_sweep.sweep_number)
                isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
                    lsq_sweeps, lsq_features, lsq_stim_timing_dict, exclude_sweep_numbers=exclude_sweeps_for_isi)
                result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_stim_timing_dict)


        # Calculate waveforms from each type of sweep - if multiple sweeps, use the earliest
        ap_v_list = []
        ap_dv_list = []

        if ap_waveforms["ssq"]["use"]:
            spiking_ssq_sweep_list = [ssq_sweeps.sweeps[swp_ind]
                for swp_ind in ssq_features["common_amp_sweeps"].index]
            spiking_ssq_info_list = [ssq_features["spikes_set"][swp_ind]
                for swp_ind in ssq_features["common_amp_sweeps"].index]
            ssq_ap_v, ssq_ap_dv = fv.first_ap_vectors(spiking_ssq_sweep_list[:1],
                spiking_ssq_info_list[:1],
                target_sampling_rate=target_sampling_rate,
                window_length=ap_waveforms["ssq"]["duration"],
                skip_clipped=True)
            ap_v_list.append(ssq_ap_v)
            ap_dv_list.append(ssq_ap_dv)

        if ap_waveforms["lsq"]["use"]:
            rheo_ind = lsq_features["rheobase_sweep"].name
            sweep = lsq_sweeps.sweeps[rheo_ind]
            lsq_ap_v, lsq_ap_dv = fv.first_ap_vectors([sweep],
                [lsq_features["spikes_set"][rheo_ind]],
                target_sampling_rate=target_sampling_rate,
                window_length=ap_waveforms["lsq"]["duration"])
            ap_v_list.append(lsq_ap_v)
            ap_dv_list.append(lsq_ap_dv)

        if extract_from_ramp and ap_waveforms["ramp"]["use"]:
            spiking_ramp_sweep_list = [ramp_sweeps.sweeps[swp_ind]
                for swp_ind in ramp_features["spiking_sweeps"].index]
            spiking_ramp_info_list = [ramp_features["spikes_set"][swp_ind]
                for swp_ind in ramp_features["spiking_sweeps"].index]
            ramp_ap_v, ramp_ap_dv = fv.first_ap_vectors(spiking_ramp_sweep_list[:1],
                spiking_ramp_info_list[:1],
                target_sampling_rate=target_sampling_rate,
                window_length=ap_waveforms["ramp"]["duration"],
                skip_clipped=True)
            ap_v_list.append(ramp_ap_v)
            ap_dv_list.append(ramp_ap_dv)

        # Combine so that differences can be assessed by analyses like sPCA
        result["first_ap_v"] = np.hstack(ap_v_list)
        result["first_ap_dv"] = np.hstack(ap_dv_list)

        target_amplitudes = np.arange(0, 100, 10)
        supra_info_list, supra_sweep_numbers = fv.identify_suprathreshold_spike_info(
            lsq_features,
            target_amplitudes,
            sweep_numbers=[swp.sweep_number for swp in lsq_sweeps.sweeps],
            shift=None,
            amp_tolerance=amp_tolerance,
            needed_amplitudes=needed_amplitudes
        )

        supra_lsq_stim_timing_list = [lsq_stim_timing_dict[sn] if sn is not None else None for sn in supra_sweep_numbers]

        actual_amps = [int(a) for a, si in zip(target_amplitudes, supra_info_list) if si is not None]
        actual_rheobase_i = int(lsq_features["rheobase_i"])

        result["long_squares_data_info"] = {"rheobase_i": actual_rheobase_i, "amplitudes_with_data": actual_amps}

        result["psth"] = fv.psth_vector(supra_info_list, supra_lsq_stim_timing_list)
        result["inst_freq"] = fv.inst_freq_vector(supra_info_list, supra_lsq_stim_timing_list)

        spike_feature_list = [
            "upstroke_downstroke_ratio",
            "peak_v",
            "fast_trough_v",
            "threshold_v",
            "width",
        ]
        for feature in spike_feature_list:
            result["spiking_" + feature] = fv.spike_feature_vector(feature,
                supra_info_list, supra_lsq_stim_timing_list)
    except Exception as detail:
        logging.warning("Exception when processing specimen {:d}".format(specimen_id))
        logging.warning(detail)
        return {"error": {"type": "processing", "details": traceback.format_exc(limit=None)}, "specimen_id": specimen_id}

    logging.info(f"Successfully processed {specimen_id}")

    # Flush the LRU cache for the data_set object
    if hasattr(data_set, "_data") and hasattr(data_set._data, "_get_series"):
        data_set._data._get_series.cache_clear()

    return result


def run_feature_vector_extraction(
        specimen_ids,
        output_dir,
        output_code,
        sweep_qc_option,
        file_list,
        sweep_qc_record_df,
        manual_fail_sweep_dict,
        extract_from_ramp,
        amp_tolerance,
        additional_fvs,
        extend_durations,
        ap_waveforms,
        needed_amplitudes,
        run_parallel=True,
    ):
    """
    Extract feature vectors from a list of cells and save results
    """

    get_data_partial = partial(data_for_specimen_id,
                               sweep_qc_option=sweep_qc_option,
                               needed_amplitudes=needed_amplitudes,
                               amp_tolerance=amp_tolerance,
                               ap_waveforms=ap_waveforms,
                               extend_durations=extend_durations,
                               extract_from_ramp=extract_from_ramp,
                               additional_fvs=additional_fvs,
                               file_list=file_list,
                               sweep_qc_record=sweep_qc_record_df,
                               manual_fail_sweeps=manual_fail_sweep_dict)

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))
    if run_parallel:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            results = executor.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)
    for r in results:
        if "error" in r:
            print(r)
        else:
            print("OK", r["id"])

def main(args):
    ids = np.genfromtxt(args["specimen_id_file"], dtype=int).tolist()

    nwb_path_file = args["nwb_path_file"]
    if nwb_path_file is None:
        file_list = lq.get_nwb_file_paths_for_specimen_ids(ids)
    else:
        with open(nwb_path_file, "r") as f:
            file_list = json.load(f)

    sweep_qc_record_file = args["sweep_qc_record_file"]
    if sweep_qc_record_file is None:
        sweep_qc_record = lq.get_sweep_states_and_tags_for_specimens(ids)
        sweep_qc_record_df = pd.DataFrame(sweep_qc_record)
        sweep_qc_record_df["tag_name"] = sweep_qc_record_df["tag_name"].fillna("None")
    else:
        sweep_qc_record_df = pd.read_csv(sweep_qc_record_file)

    manual_fail_sweep_file = args["manual_fail_sweep_file"]
    if manual_fail_sweep_file is not None:
        manual_fail_df = pd.read_csv(manual_fail_sweep_file)
        manual_fail_sweep_dict = {}
        for specimen_id in manual_fail_df.specimen_id.unique():
            sweeps_for_specimen = manual_fail_df.loc[manual_fail_df.specimen_id == specimen_id, "sweep_number"].tolist()
            manual_fail_sweep_dict[specimen_id] = sweeps_for_specimen
    else:
        manual_fail_sweep_dict = None

    run_feature_vector_extraction(
        specimen_ids=ids,
        output_dir=args["output_dir"],
        output_code=args["output_code"],
        sweep_qc_option=args["sweep_qc_option"],
        file_list=file_list,
        sweep_qc_record_df=sweep_qc_record_df,
        manual_fail_sweep_dict=manual_fail_sweep_dict,
        extract_from_ramp=args["extract_from_ramp"],
        amp_tolerance=args["amp_tolerance"],
        additional_fvs=args["additional_fvs"],
        extend_durations=args["extend_durations"],
        ap_waveforms=args["ap_waveforms"],
        needed_amplitudes=args["needed_amplitudes"],
        run_parallel=args["run_parallel"],
    )

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CollectFeatureVectorParameters)
    main(module.args)
