import numpy as np
import logging
import pandas as pd
from . import stim_features as stf
from . import data_set_features as dsf
from . import stimulus_protocol_analysis as spa
import error as er



def extract_feature_vectors(data_set,
                            ramp_sweep_numbers,
                            short_square_sweep_numbers,
                            long_square_sweep_numbers):
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
    if len(long_square_sweep_numbers) == 0:
        raise er.FeatureError("No long_square sweeps available for feature extraction")

    lsq_sweeps = data_set.sweep_set(long_square_sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(lsq_sweeps.sweeps[0].i, lsq_sweeps.sweeps[0].t)
    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                  start = lsq_start,
                                                  end = lsq_start+lsq_dur,
                                                  **dsf.detection_parameters(data_set.LONG_SQUARE))
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)
    lsq_features = lsq_an.analyze(lsq_sweeps)

    # short squares
    if len(short_square_sweep_numbers) == 0:
        raise er.FeatureError("No short square sweeps available for feature extraction")

    ssq_sweeps = data_set.sweep_set(short_square_sweep_numbers)

    ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(ssq_sweeps.sweeps[0].i, ssq_sweeps.sweeps[0].t)
    ssq_spx, ssq_spfx = dsf.extractors_for_sweeps(ssq_sweeps,
                                              est_window = [ssq_start,ssq_start+0.001],
                                              **dsf.detection_parameters(data_set.SHORT_SQUARE))
    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)
    ssq_features = ssq_an.analyze(ssq_sweeps)

    # ramps
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

    all_features = feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                                   lsq_features, ssq_features, ramp_features)
    return all_features


def feature_vectors(lsq_sweeps, ssq_sweeps, ramp_sweeps,
                    lsq_features, ssq_features, ramp_features,
                    feature_width=20, rate_width=50):
    """Feature vectors from stimulus set features"""

    results = {}
    result["step_subthresh"] = step_subthreshold(lsq_features)
    result["spiking"] = spiking_features(lsq_features, feature_width, rate_width)
    result["isi_shape"] = isi_shape(lsq_features)
    result["subthresh_norm"] = subthresh_norm(lsq_features)

    result["first_ap"] = first_ap_features(lsq_features, ssq_features, ramp_features)

    return result
