import numpy as np
from ipfx.sweep import Sweep, SweepSet
import ipfx.feature_vectors as fv

from neuroanalysis.miesnwb import MiesNwb
import argschema as ags
import os
import json
import logging


class SynPhysFeatureVectorSchema(ags.ArgSchema):
    nwb_file = ags.fields.InputFile(default="/allen/programs/celltypes/workgroups/ivscc/nathang/synphys_ephys/data/2018_11_06_140408-compressed.nwb")
    output_dir = ags.fields.OutputDir(default="/allen/programs/celltypes/workgroups/ivscc/nathang/fv_output/")
    project = ags.fields.String(default="SynPhys")


class MPSweep(Sweep):
    """Adapter for neuroanalysis.Recording => ipfx.Sweep
    """
    def __init__(self, rec):
        pri = rec['primary']
        cmd = rec['command']
        t = pri.time_values
        v = pri.data * 1e3  # convert to mV
        holding = rec.stimulus.items[0].amplitude  # todo: select holding item explicitly; don't assume it is [0]
        i = (cmd.data - holding) * 1e12   # convert to pA with holding current removed
        srate = pri.sample_rate
        sweep_num = rec.parent.key
        clamp_mode = rec.clamp_mode  # this will be 'ic' or 'vc'; not sure if that's right

        Sweep.__init__(self, t, v, i,
                       clamp_mode=clamp_mode,
                       sampling_rate=srate,
                       sweep_number=sweep_num,
                       epochs=None)


def recs_to_sweeps(recs, min_pulse_dur=np.inf):
    sweeps = []
    for rec in recs:
        # get square pulse start/stop times
        pulse = rec.stimulus.items[3]  # this assumption is bound to fail sooner or later.
        start = pulse.global_start_time
        end = start + pulse.duration

        # pulses may have different start times, so we shift time values to make all pulses start at t=0
        rec['primary'].t0 = -start
        # pulses may have different durations as well, so we just use the smallest duration
        min_pulse_dur = min(min_pulse_dur, end-start)

        sweep = MPSweep(rec)
        sweeps.append(sweep)
    return sweeps, min_pulse_dur


def main(nwb_file, output_dir, project, **kwargs):
    nwb = MiesNwb(nwb_file)


    # SPECIFICS FOR EXAMPLE NWB =========

    # Only analyze one channel at a time
    channel = 0

    # We can work out code to automatically extract these based on stimulus names later.
    if_sweep_inds = [39, 45]
    targetv_sweep_inds = [15, 21]

    # END SPECIFICS =====================


    # Assemble all Recordings and convert to Sweeps
    supra_sweep_ids = list(range(*if_sweep_inds))
    sub_sweep_ids = list(range(*targetv_sweep_inds))

    supra_recs = [nwb.contents[i][channel] for i in supra_sweep_ids]
    sub_recs = [nwb.contents[i][channel] for i in sub_sweep_ids]

    # Build sweep sets
    lsq_supra_sweep_list, lsq_supra_dur = recs_to_sweeps(supra_recs)
    lsq_sub_sweep_list, lsq_sub_dur = recs_to_sweeps(sub_recs)
    lsq_supra_sweeps = SweepSet(lsq_supra_sweep_list)
    lsq_sub_sweeps = SweepSet(lsq_sub_sweep_list)

    lsq_supra_start = 0
    lsq_supra_end = lsq_supra_dur
    lsq_sub_start = 0
    lsq_sub_end = lsq_sub_dur

    # Pre-process sweeps
    lsq_supra_spx, lsq_supra_spfx = dsf.extractors_for_sweeps(lsq_supra_sweeps, start=lsq_supra_start, end=lsq_supra_end)
    lsq_supra_an = spa.LongSquareAnalysis(lsq_supra_spx, lsq_supra_spfx, subthresh_min_amp=-100., require_subthreshold=False)
    lsq_supra_features = lsq_supra_an.analyze(lsq_supra_sweeps)

    lsq_sub_spx, lsq_sub_spfx = dsf.extractors_for_sweeps(lsq_sub_sweeps, start=lsq_sub_start, end=lsq_sub_end)
    lsq_sub_an = spa.LongSquareAnalysis(lsq_sub_spx, lsq_sub_spfx, subthresh_min_amp=-100., require_suprathreshold=False)
    lsq_sub_features = lsq_sub_an.analyze(lsq_sub_sweeps)

    # Calculate feature vectors
    result = {}
    (subthresh_hyperpol_dict,
    hyperpol_deflect_dict) = fv.identify_subthreshold_hyperpol_with_amplitudes(lsq_sub_features,
        lsq_sub_sweeps)
    target_amps_for_step_subthresh = [-90, -70, -50, -30, -10]
    result["step_subthresh"] = fv.step_subthreshold(
        subthresh_hyperpol_dict, target_amps_for_step_subthresh,
        lsq_sub_start, lsq_sub_end, amp_tolerance=5)
    result["subthresh_norm"] = fv.subthresh_norm(subthresh_hyperpol_dict, hyperpol_deflect_dict,
        lsq_sub_start, lsq_sub_end)

    (subthresh_depol_dict,
    depol_deflect_dict) = fv.identify_subthreshold_depol_with_amplitudes(lsq_supra_features,
        lsq_supra_sweeps)
    result["subthresh_depol_norm"] = fv.subthresh_depol_norm(subthresh_depol_dict,
        depol_deflect_dict, lsq_supra_start, lsq_supra_end)
    isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
        lsq_supra_sweeps, lsq_supra_features, lsq_supra_end - lsq_supra_start)
    result["isi_shape"] = fv.isi_shape(isi_sweep, isi_sweep_spike_info, lsq_supra_end)

    # Calculate AP waveform from long squares
    rheo_ind = lsq_supra_features["rheobase_sweep"].name
    sweep = lsq_supra_sweeps.sweeps[rheo_ind]
    lsq_ap_v, lsq_ap_dv = fv.first_ap_vectors([sweep],
        [lsq_supra_features["spikes_set"][rheo_ind]],
        window_length=ap_window_length)

    result["first_ap_v"] = lsq_ap_v
    result["first_ap_dv"] = lsq_ap_dv

    target_amplitudes = np.arange(0, 120, 20)
    supra_info_list = fv.identify_suprathreshold_sweep_sequence(
        lsq_supra_features, target_amplitudes, shift=10)
    result["psth"] = fv.psth_vector(supra_info_list, lsq_supra_start, lsq_supra_end)
    result["inst_freq"] = fv.inst_freq_vector(supra_info_list, lsq_supra_start, lsq_supra_end)
    spike_feature_list = [
        "upstroke_downstroke_ratio",
        "peak_v",
        "fast_trough_v",
        "threshold_v",
        "width",
    ]
    for feature in spike_feature_list:
        result["spiking_" + feature] = fv.spike_feature_vector(feature,
            supra_info_list, lsq_supra_start, lsq_supra_end)

    # Save the results
    specimen_ids = [0]
    results = [result]

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
        if len(data.shape) == 1: # it'll be 1D if there's just one specimen
            data = np.reshape(data, (1, -1))
        if data.shape[0] < len(used_ids):
            logging.warn("Missing data!")
            missing = np.array([k not in r for r in results])
            print(k, np.array(used_ids)[missing])
        np.save(os.path.join(output_dir, "fv_{:s}_{:s}.npy".format(k, project)), data)

    np.save(os.path.join(output_dir, "fv_ids_{:s}.npy".format(project)), used_ids)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SynPhysFeatureVectorSchema)
    main(**module.args)
