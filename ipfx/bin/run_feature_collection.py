import numpy as np
import pandas as pd
import scipy
import argschema as ags
import ipfx.lims_queries as lq
import ipfx.feature_vectors as fv
import ipfx.script_utils as su
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
import logging
from multiprocessing import Pool
from functools import partial


class CollectFeatureParameters(ags.ArgSchema):
    output_file = ags.fields.OutputFile(default=None)
    input = ags.fields.InputFile(default=None, allow_none=True)
    project = ags.fields.String(default="T301")
    include_failed_sweeps = ags.fields.Boolean(default=False)
    include_failed_cells = ags.fields.Boolean(default=False)
    run_parallel = ags.fields.Boolean(default=True)
    data_source = ags.fields.String(
        description="Source of NWB files ('sdk' or 'lims' or 'filesystem')",
        default="sdk",
        validate=lambda x: x in ["sdk", "lims", "filesystem"]
        )


def data_for_specimen_id(specimen_id, passed_only, data_source, ontology, file_list=None):
    data_set = su.dataset_for_specimen_id(specimen_id, data_source, ontology, file_list)
    if type(data_set) is dict and "error" in data_set:
        logging.warning("Problem getting AibsDataSet for specimen {:d} from LIMS".format(specimen_id))
        return {}

    try:
        lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.long_square_names)
        ssq_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.short_square_names)
        ramp_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.ramp_names)
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
#         return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=1)}}
        return {}

    try:
        result = extract_features(data_set, ramp_sweep_numbers, ssq_sweep_numbers, lsq_sweep_numbers)
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
#         return {"error": {"type": "processing", "details": traceback.format_exc(limit=1)}}
        return {}

    result["specimen_id"] = specimen_id
    return result


def extract_features(data_set, ramp_sweep_numbers, ssq_sweep_numbers, lsq_sweep_numbers,
                     amp_interval=20, max_above_rheo=100):
    features = {}
    # RAMP FEATURES -----------------
    if len(ramp_sweep_numbers) > 0:
        _, _, ramp_an = su.preprocess_ramp_sweeps(data_set,
            ramp_sweep_numbers)
        first_spike_ramp_features = first_spike_ramp(ramp_an)
        features.update(first_spike_ramp_features)

    # SHORT SQUARE FEATURES -----------------
    if len(ssq_sweep_numbers) > 0:
        _, basic_ssq_features, ssq_an = su.preprocess_short_square_sweeps(data_set,
            ssq_sweep_numbers)
        first_spike_ssq_features = first_spike_ssq(ssq_an)
        first_spike_ssq_features["short_square_current"] = basic_ssq_features["stimulus_amplitude"]
        features.update(first_spike_ssq_features)

    # LONG SQUARE SUBTHRESHOLD FEATURES -----------------
    if len(lsq_sweep_numbers) > 0:
        (lsq_sweeps,
        basic_lsq_features,
        lsq_an,
        lsq_start,
        lsq_end) = su.preprocess_long_square_sweeps(data_set, lsq_sweep_numbers)

        features.update({
            "input_resistance": basic_lsq_features["input_resistance"],
            "tau": basic_lsq_features["tau"],
            "v_baseline": basic_lsq_features["v_baseline"],
            "sag_nearest_minus_100": basic_lsq_features["sag"],
            "sag_measured_at": basic_lsq_features["vm_for_sag"],
            "rheobase_i": int(basic_lsq_features["rheobase_i"]),
            "fi_linear_fit_slope": basic_lsq_features["fi_fit_slope"],
        })

        # TODO (maybe): port sag_from_ri code over

        # Identify suprathreshold set for analysis
        sweep_table = basic_lsq_features["spiking_sweeps"]
        mask_supra = sweep_table["stim_amp"] >= basic_lsq_features["rheobase_i"]
        sweep_indexes = fv._consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
        amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - basic_lsq_features["rheobase_i"])
        spike_data = np.array(basic_lsq_features["spikes_set"])

        for amp, swp_ind in zip(amps, sweep_indexes):
            if (amp % amp_interval != 0) or (amp > max_above_rheo) or (amp < 0):
                continue
            amp_label = int(amp / amp_interval)

            first_spike_lsq_sweep_features = first_spike_lsq(spike_data[swp_ind])
            features.update({"ap_1_{:s}_{:d}_long_square".format(f, amp_label): v
                             for f, v in first_spike_lsq_sweep_features.items()})

            mean_spike_lsq_sweep_features = mean_spike_lsq(spike_data[swp_ind])
            features.update({"ap_mean_{:s}_{:d}_long_square".format(f, amp_label): v
                             for f, v in mean_spike_lsq_sweep_features.items()})

            sweep_feature_list = [
                "first_isi",
                "avg_rate",
                "isi_cv",
                "latency",
                "median_isi",
                "adapt",
            ]

            features.update({"{:s}_{:d}_long_square".format(f, amp_label): sweep_table.at[swp_ind, f]
                             for f in sweep_feature_list})
            features["stimulus_amplitude_{:d}_long_square".format(amp_label)] = int(amp + basic_lsq_features["rheobase_i"])

        rates = sweep_table.loc[sweep_indexes, "avg_rate"].values
        features.update(fi_curve_fit(amps, rates))

    return features


def first_spike_ramp(ramp_analyzer,
                     feature_list = [
                        "threshold_v",
                        "peak_v",
                        "upstroke",
                        "downstroke",
                        "upstroke_downstroke_ratio",
                        "width",
                        "fast_trough_v",
                    ]):
    first_spike_features = ramp_analyzer.mean_features_first_spike(ramp_analyzer._spikes_set, feature_list)
    return {"ap_1_" + f + "_ramp": v for f, v in first_spike_features.items()}


def first_spike_ssq(ssq_analyzer,
                    feature_list = [
                        "threshold_v",
                        "peak_v",
                        "upstroke",
                        "downstroke",
                        "upstroke_downstroke_ratio",
                        "width",
                        "fast_trough_v",
                    ]):
    first_spike_features = ssq_analyzer.mean_features_first_spike(ssq_analyzer._spikes_set, feature_list)
    return {"ap_1_" + f + "_short_square": v for f, v in first_spike_features.items()}


def first_spike_lsq(spike_data,
                    feature_list = [
                        "threshold_v",
                        "peak_v",
                        "upstroke",
                        "downstroke",
                        "upstroke_downstroke_ratio",
                        "width",
                        "fast_trough_v",
                    ]):
    return {f: spike_data.loc[0, f] for f in feature_list}


def mean_spike_lsq(spike_data,
                   feature_list = [
                       "threshold_v",
                       "peak_v",
                       "upstroke",
                       "downstroke",
                       "upstroke_downstroke_ratio",
                       "width",
                       "fast_trough_v",
                   ]):

    return {f: spike_data[f].mean(skipna=True) for f in feature_list}


def fi_curve_fit(amps, rates):
    result = {
        "fi_sqrt_fit_A": np.nan,
        "fi_sqrt_fit_Ic": np.nan,
    }

    min_sweeps = 3

    if len(amps) < min_sweeps:
        return result

    guess = (1., amps[0])

    try:
        popt, pcov = scipy.optimize.curve_fit(sqrt_curve, amps, rates,
                                              p0=guess)
        if np.isinf(pcov[0, 0]):
            logging.info("Nonlinear curve fit of f-I curve produced infinite covariance")
            popt = lin_sqrt_fit(amps, rates)
    except RuntimeError as e:
        logging.debug("Nonlinear curve fit of f-I curve failed - using fallback linear method")
        if np.all(rates[0] == rates): # All values are the same
            popt = (0., np.min(amps))
        else:
            popt = lin_sqrt_fit(amps, rates)

    for k, v in zip(["fi_sqrt_fit_A", "fi_sqrt_fit_Ic"], popt):
        result[k] = v

    return result


def sqrt_curve(x, A, Ic):
    vals = x - Ic
    vals[x < Ic] = 0.
    out = A * np.sqrt(vals)
    return out


def lin_sqrt_fit(x, y):
    b = y ** 2
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = scipy.linalg.lstsq(A, b)[0]

    return np.array((np.sqrt(m), -c / m))


def run_feature_collection(ids=None, project="T301", include_failed_sweeps=True, include_failed_cells=False,
         output_file="", run_parallel=True, data_source="lims", file_list=None, **kwargs):
    if ids is not None:
        specimen_ids = ids
    else:
        specimen_ids = lq.project_specimen_ids(project, passed_only=not include_failed_cells)

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))

    ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    get_data_partial = partial(data_for_specimen_id,
                               passed_only=not include_failed_sweeps,
                               data_source=data_source,
                               ontology=ontology,
                               file_list=file_list)

    if run_parallel:
        pool = Pool()
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    df = pd.DataFrame([r for r in results if len(r) > 0])
    logging.info("shape {}".format(df.shape))
    df.set_index("specimen_id").to_csv(output_file)


def main():

    module = ags.ArgSchemaParser(schema_type=CollectFeatureParameters)

    if module.args["input"]: # input file should be list of IDs on each line
        with open(module.args["input"], "r") as f:
            ids = [int(line.strip("\n")) for line in f]
        run_feature_collection(ids=ids, **module.args)
    else:
        run_feature_collection(**module.args)

if __name__ == "__main__": main()
