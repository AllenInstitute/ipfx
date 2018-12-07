import numpy as np
import logging
import pandas as pd
from . import stim_features as stf
from . import data_set_features as dsf
from . import stimulus_protocol_analysis as spa
from . import time_series_utils as tsu
import error as er



def extract_feature_vectors(data_set,
                            ramp_sweep_numbers,
                            short_square_sweep_numbers,
                            long_square_sweep_numbers,
                            use_lsq=True, use_ssq=True, use_ramp=True):
    """Extract feature vectors for downstream dimensionality reduction

    Parameters
    ----------
    data_set : AibsDataSet
    ramp_sweep_numbers :
    short_square_sweep_numbers :
    long_square_sweep_numbers :

    Returns
    -------
    all_features : dictionary of feature vectors
    """

    # long squares
    if use_lsq:
        if len(long_square_sweep_numbers) == 0:
            raise er.FeatureError("No long_square sweeps available for feature extraction")

        check_lsq_sweeps = data_set.sweep_set(long_square_sweep_numbers)
        lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(check_lsq_sweeps.sweeps[0].i, check_lsq_sweeps.sweeps[0].t)

        # Check that all sweeps are long enough
        extra_dur = 0.2
        good_lsq_sweep_numbers = [n for n, s in zip(long_square_sweep_numbers, check_lsq_sweeps.sweeps)
                                  if s.t[-1] >= lsq_start + lsq_dur + extra_dur]
        lsq_sweeps = data_set.sweep_set(good_lsq_sweep_numbers)

        lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                      start = lsq_start,
                                                      end = lsq_start + lsq_dur,
                                                      **dsf.detection_parameters(data_set.LONG_SQUARE))
        lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)
        lsq_features = lsq_an.analyze(lsq_sweeps)
    else:
        lsq_sweeps = None
        lsq_features = None

    # short squares
    if use_ssq:
        if len(short_square_sweep_numbers) == 0:
            raise er.FeatureError("No short square sweeps available for feature extraction")

        ssq_sweeps = data_set.sweep_set(short_square_sweep_numbers)

        ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(ssq_sweeps.sweeps[0].i, ssq_sweeps.sweeps[0].t)
        ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                                      est_window = [ssq_start, ssq_start+0.001],
                                                      **dsf.detection_parameters(data_set.SHORT_SQUARE))
        ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
        ssq_features = ssq_an.analyze(ssq_sweeps)
    else:
        ssq_sweeps = None
        ssq_features = None

    # ramps
    if use_ramp:
        if len(ramp_sweep_numbers) == 0:
            raise er.FeatureError("No ramp sweeps available for feature extraction")

        ramp_sweeps = data_set.sweep_set(ramp_sweep_numbers)

        ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)
        logging.info("Ramp stim %f, %f", ramp_start, ramp_dur)

        ramp_spx, ramp_spfx = dsf.extractors_for_sweeps(ramp_sweeps,
                                                    start = ramp_start,
                                                    **dsf.detection_parameters(data_set.RAMP))
        ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
        ramp_features = ramp_an.analyze(ramp_sweeps)
    else:
        ramp_sweeps = None
        ramp_features = None

    all_features = feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                                   lsq_features, ssq_features, ramp_features,
                                   lsq_start, lsq_start + lsq_dur)
    return all_features


def feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                    lsq_features, ssq_features, ramp_features,
                    lsq_start, lsq_end,
                    feature_width=20, rate_width=50):
    """Feature vectors from stimulus set features"""

    result = {}
    result["step_subthresh"] = step_subthreshold(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    result["subthresh_norm"] = subthresh_norm(lsq_sweeps, lsq_features, lsq_start, lsq_end)
    return result

    result["spiking"] = spiking_features(lsq_features, feature_width, rate_width)
    result["isi_shape"] = isi_shape(lsq_features)

    result["first_ap"] = first_ap_features(lsq_features, ssq_features, ramp_features)

    return result


def step_subthreshold(sweep_set, features, start, end,
                      extend_duration=0.2, subsample_interval=0.01,
                      low_amp=-90., high_amp=10., amp_step=20.):
    """ Subsample set of subthreshold step responses including regions before and after step

        Parameters
        ----------
        sweep_set : SweepSet
        features : output of LongSquareAnalysis.analyze()
        start : stimulus interval start (seconds)
        end : stimulus interval end (seconds)
        extend_duration : in seconds (default 0.2)
        subsample_interval : in seconds (default 0.01)

        Returns
        -------
        output_vector : subsampled, concatenated voltage trace
    """
    subthresh_df = features["subthreshold_membrane_property_sweeps"]
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweep_set.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)

    subthresh_data = {}
    for amp, swp in zip(subthresh_amps, subthresh_sweeps):
        start_index = tsu.find_time_index(swp.t, start - extend_duration)
        delta_t = swp.t[1] - swp.t[0]
        subsample_width = int(np.round(subsample_interval / delta_t))
        end_index = tsu.find_time_index(swp.t, end + extend_duration)
        subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
        subthresh_data[amp] = subsampled_v

    extend_length = int(np.round(extend_duration / subsample_interval))
    use_amps = np.arange(low_amp, high_amp, amp_step)
    n_individual = len(subsampled_v)
    neighbor_amps = sorted([a for a in subthresh_amps if a >= low_amp and a <= high_amp])
    output_vector = np.zeros(len(use_amps) * n_individual)
    for i, amp in enumerate(use_amps):
        if amp in subthresh_data:
            output_vector[i * n_individual:(i + 1) * n_individual] = subthresh_data[amp]
        else:
            lower_amp = 0
            upper_amp = 0
            for a in neighbor_amps:
                if a < amp:
                    if lower_amp == 0:
                        lower_amp = a
                    elif a > lower_amp:
                        lower_amp = a
                if a > amp:
                    if upper_amp == 0:
                        upper_amp = a
                    elif a < upper_amp:
                        upper_amp = a

            avg = np.zeros_like(subsampled_v)
            if lower_amp != 0 and upper_amp != 0:
                avg = (subthresh_data[lower_amp] + subthresh_data[upper_amp]) / 2.
                scale = amp / ((lower_amp + upper_amp) / 2.)
                base_v = avg[:extend_length].mean()
                avg[extend_length:-extend_length] = (avg[extend_length:-extend_length] - base_v) * scale + base_v
            elif lower_amp != 0:
                avg = subthresh_data[lower_amp].copy()
                scale = amp / lower_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v
            elif upper_amp != 0:
                avg = subthresh_data[upper_amp].copy()
                scale = amp / upper_amp
                base_v = avg[:extend_length].mean()
                avg[extend_length:] = (avg[extend_length:] - base_v) * scale + base_v

            output_vector[i * n_individual:(i + 1) * n_individual] = avg

    return output_vector


def subsample_average(x, width):
    """Downsamples x by averaging `width` points"""

    avg = np.nanmean(x.reshape(-1, width), axis=1)
    return avg


def subthresh_norm(sweep_set, features, start, end,
                   extend_duration=0.2, subsample_interval=0.01):
    """ Largest amplitude subthreshold step response normalized to baseline and peak deflection

        Parameters
        ----------
            sweep_set : SweepSet
            features : output of LongSquareAnalysis.analyze()
            start : stimulus interval start (seconds)
            end : stimulus interval end (seconds)
            extend_duration: in seconds (default 0.2)
            subsample_interval: in seconds (default 0.01)

        Returns
        -------
        subsampled_v : subsampled, normalized voltage trace
    """
    subthresh_df = features["subthreshold_membrane_property_sweeps"]
    subthresh_sweep_ind = subthresh_df.index.tolist()
    subthresh_sweeps = np.array(sweep_set.sweeps)[subthresh_sweep_ind]
    subthresh_amps = np.rint(subthresh_df["stim_amp"].values)

    # PRE QC ISSUE: Check for bad amps; shouldn't have to do this in general
    subthresh_amps[subthresh_amps < -1000] = np.inf

    min_sweep_ind = np.argmin(subthresh_amps)
    swp = subthresh_sweeps[min_sweep_ind]
    base = subthresh_df.at[subthresh_df.index[min_sweep_ind], "v_baseline"]
    deflect_v, deflect_ind = subthresh_df.at[subthresh_df.index[min_sweep_ind], "peak_deflect"]
    delta = base - deflect_v

    start_index = tsu.find_time_index(swp.t, start - extend_duration)
    delta_t = swp.t[1] - swp.t[0]
    subsample_width = int(np.round(subsample_interval / delta_t))
    end_index = tsu.find_time_index(swp.t, end + extend_duration)
    subsampled_v = subsample_average(swp.v[start_index:end_index], subsample_width)
    subsampled_v -= base
    subsampled_v /= delta

    return subsampled_v

