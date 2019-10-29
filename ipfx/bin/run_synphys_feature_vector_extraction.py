import numpy as np
from ipfx.ephys_data_set import SweepSet
import ipfx.feature_vectors as fv
import ipfx.synphys as synphys
import ipfx.data_set_features as dsf

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
        cell = synphys.cell_from_mpid(mpid)
        # data is a MiesNwb instance
        nwb = cell.experiment.data
        channel = cell.electrode.device_id
        sweeps_dict = synphys.sweeps_dict_from_cell(cell)
        supra_sweep_ids = sweeps_dict['If_Curve_DA_0']
        sub_sweep_ids = sweeps_dict['TargetV_DA_0']
        lsq_supra_sweep_list = [synphys.MPSweep(nwb.contents[i][channel]) for i in supra_sweep_ids]
        lsq_sub_sweep_list = [synphys.MPSweep(nwb.contents[i][channel]) for i in sub_sweep_ids]
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
        logging.warn("Exception when getting sweeps for specimen {}".format(mpid))
        logging.warn(detail)
        return {"error": {"type": "dataset", "details": traceback.format_exc(limit=None)}}

    result = {}
    try:
        # Process hyperpolarizing long square features
        sub_start = 0.
        sub_end = sub_start + lsq_sub_dur

        sub_spx, sub_spfx = dsf.extractors_for_sweeps(
            lsq_sub_sweeps,
            start=sub_start,
            end=sub_end,
            min_peak=-25,
            **dsf.detection_parameters(data_set.LONG_SQUARE)
        )
        sub_an = spa.LongSquareAnalysis(sub_spx, sub_spfx,
            subthresh_min_amp=-100.)
        sub_features = lsq_an.analyze(lsq_sub_sweeps)

        # Process depolarizing long square features
        supra_start = 0.
        supra_end = supra_start + lsq_supra_dur

        supra_spx, supra_spfx = dsf.extractors_for_sweeps(
            lsq_supra_sweeps,
            start=supra_start,
            end=supra_end,
            min_peak=-25,
            **dsf.detection_parameters(data_set.LONG_SQUARE)
        )
        supra_an = spa.LongSquareAnalysis(supra_spx, supra_spfx,
            subthresh_min_amp=-100.)
        supra_features = lsq_an.analyze(lsq_supra_sweeps)

        # Extract hyperpolarizing step-related feature vectors
        (subthresh_hyperpol_dict,
        hyperpol_deflect_dict) = fv.identify_subthreshold_hyperpol_with_amplitudes(sub_features,
            lsq_sub_sweeps)
        target_amps_for_step_subthresh = [-90, -70, -50, -30, -10]
        result["step_subthresh"] = fv.step_subthreshold(
            subthresh_hyperpol_dict, target_amps_for_step_subthresh,
            sub_start, sub_end, amp_tolerance=9.9)
        result["subthresh_norm"] = fv.subthresh_norm(subthresh_hyperpol_dict, hyperpol_deflect_dict,
            sub_start, sub_end)

        # Extract depolarizing step-related feature vectors
        (subthresh_depol_dict,
        depol_deflect_dict) = fv.identify_subthreshold_depol_with_amplitudes(supra_features,
            lsq_supra_sweeps)
        result["subthresh_depol_norm"] = fv.subthresh_depol_norm(subthresh_depol_dict,
            depol_deflect_dict, supra_start, supra_end)
        isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
            lsq_supra_sweeps, supra_features, supra_end - supra_start)
        result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, supra_end)

        # Calculate waveform from rheobase sweep
        rheo_ind = supra_features["rheobase_sweep"].name
        sweep = lsq_supra_sweeps.sweeps[rheo_ind]
        lsq_ap_v, lsq_ap_dv = fv.first_ap_vectors([sweep],
            [supra_features["spikes_set"][rheo_ind]],
            target_sampling_rate=target_sampling_rate,
            window_length=ap_window_length)

        # Combine so that differences can be assessed by analyses like sPCA
        result["first_ap_v"] = np.hstack([np.zeros_like(lsq_ap_v), lsq_ap_v, np.zeros_like(lsq_ap_v)])
        result["first_ap_dv"] = np.hstack([np.zeros_like(lsq_ap_dv), lsq_ap_dv, np.zeros_like(lsq_ap_dv)])

        target_amplitudes = np.arange(0, 120, 20)
        supra_info_list = fv.identify_suprathreshold_spike_info(
            supra_features, target_amplitudes, shift=10)
        result["psth"] = fv.psth_vector(supra_info_list, supra_start, supra_end)
        result["inst_freq"] = fv.inst_freq_vector(supra_info_list, supra_start, supra_end)

        spike_feature_list = [
            "upstroke_downstroke_ratio",
            "peak_v",
            "fast_trough_v",
            "threshold_v",
            "width",
        ]
        for feature in spike_feature_list:
            result["spiking_" + feature] = fv.spike_feature_vector(feature,
                supra_info_list, supra_start, supra_end)
    except Exception as detail:
        logging.warn("Exception when processing feature vectors for specimen {}".format(mpid))
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
            print(k, np.array(used_ids)[missing])
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, project)), data)

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(project)), used_ids)


def main():
    module = ags.ArgSchemaParser(schema_type=SynPhysFeatureVectorSchema)

    # Input file should be list of acq timestamps and ext_ids separeted
    # by an '_' from the aisynphys database.  Each new cell should be on a
    # new line with an underscore in the middle.
    # For example:
    #      1490137102.372_7
    #      1490137102.372_4
    if module.args["input"]:
        with open(module.args["input"], "r") as f:
            ids = [line.strip("\n") for line in f]
        run_cells(ids=ids, **module.args)
    else:
        run_cells(**module.args)


if __name__ == "__main__": main()
