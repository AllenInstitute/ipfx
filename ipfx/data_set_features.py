# Allen Institute Software License - This software license is the 2-clause BSD
# license plus a third clause that prohibits redistribution for commercial
# purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Redistributions for commercial purposes are not permitted without the
# Allen Institute's written permission.
# For purposes of this license, commercial purposes is the incorporation of the
# Allen Institute's software into anything for which you will charge fees or
# other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
import functools
import numpy as np
from collections import defaultdict
import logging
from .feature_extractor import SpikeFeatureExtractor,SpikeTrainFeatureExtractor
from ipfx.stimulus import StimulusOntology, StimulusType, get_stimulus_type
from . import spike_features as spkf
from . import stimulus_protocol_analysis as spa
from . import stim_features as stf
from . import feature_record as fr
from . import error as er
from . import logging_utils as lu


DETECTION_PARAMETERS = defaultdict(
    lambda: {},
    {
        # To override detection parameters for specific StimulusType, add it here.
        # If not explicitly listed, default detection parameters will be used (see extractors_for_sweeps()).
        # See ipfx.stimulus for stimulus types
        StimulusType.SHORT_SQUARE: {'thresh_frac_floor': 0.1 },
        StimulusType.CHIRP: {"filter_frequency": None}
    }
)


SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS = {
    20.0: -100.0,
    40.0: -200.0
    }

TEST_PULSE_DURATION_SEC = 0.4


def detection_parameters(stimulus_type):
    return DETECTION_PARAMETERS[stimulus_type]


def detection_parameters_from_stimulus_name(stimulus_name):
    try:
        stimulus_type = get_stimulus_type(stimulus_name)
        dp = detection_parameters(stimulus_type)
    except ValueError as e:
        logging.warning(f"Warning: {e}\nUsing default detection parameters")
        dp = detection_parameters(None)
    return dp


def record_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result, errors = fn(*args, **kwargs)
        failed = (errors is not None)
        return (
            result,
            {'failed_fx': failed,
             'fail_fx_message': str(errors) if failed else None}
            )
    return wrapper


def fallback_on_error(fallback_value=None, catch_errors=(Exception)):
    def fallback_on_error_decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                error = None
            except catch_errors as e:
                logging.warning(e)
                result = fallback_value
                error = e
            return result, error
        return wrapper
    return fallback_on_error_decorator


def extractors_for_sweeps(sweep_set,
                          dv_cutoff=20., thresh_frac=0.05,
                          reject_at_stim_start_interval=0,
                          min_peak=-30,
                          filter_frequency=10.,
                          thresh_frac_floor=None,
                          est_window=None,
                          start=None, end=None):
    """Extract data from sweeps

    Parameters
    ----------
    sweep_set : SweepSet object

    dv_cutoff : float
    thresh_frac :
    reject_at_stim_start_interval :
    thresh_frac_floor
    est_window :
    start :
    end :

    Returns
    -------
    spx : SpikeExtractor object
    spfx : SpikeTrainFeatureExtractor object

    """

    if est_window is not None:
        dv_cutoff, thresh_frac = spkf.estimate_adjusted_detection_parameters(sweep_set.v, sweep_set.t,
                                                                           est_window[0],
                                                                           est_window[1])

    if thresh_frac_floor is not None:
        thresh_frac = max(thresh_frac_floor, thresh_frac)

    sfx = SpikeFeatureExtractor(dv_cutoff=dv_cutoff,
        thresh_frac=thresh_frac,
        start=start,
        end=end,
        min_peak=min_peak,
        filter=filter_frequency,
        reject_at_stim_start_interval=reject_at_stim_start_interval)

    stfx = SpikeTrainFeatureExtractor(start, end)

    return sfx, stfx


@record_errors
@fallback_on_error(fallback_value={})
def extract_sweep_features(data_set, sweep_table):
    sweep_groups = sweep_table.groupby(data_set.STIMULUS_NAME)[data_set.SWEEP_NUMBER]

    # extract sweep-level features
    lu.log_pretty_header("Analyzing sweep features:",level=2)

    sweep_features = {}

    for stimulus_name, sweep_numbers in sweep_groups:
        sweep_numbers = sorted(sweep_numbers)

        sweep_set = data_set.sweep_set(sweep_numbers)
        sweep_set.align_to_start_of_epoch("experiment")

        dp = detection_parameters_from_stimulus_name(stimulus_name).copy()
        for k in [ "start", "end" ]:
            if k in dp:
                dp.pop(k)

        sfx, _ = extractors_for_sweeps(sweep_set, **dp)

        for sn, sweep in zip(sweep_numbers, sweep_set.sweeps):
#            logging.info("Extracting features from the sweep %d" % sn)
            spikes_df = sfx.process(sweep.t, sweep.v, sweep.i)
            sweep_features[sn] = {'spikes': spikes_df.to_dict(orient='records'), "sweep_number": sn }

    return sweep_features


def select_subthreshold_min_amplitude(stim_amps, decimals=0):
    """Find the min delta between amplitudes of coarse long square sweeps.  Includes failed sweeps.

    Parameters
    ----------
    stim_amps: list of stimulus amplitudes
    decimals: int of decimals to keep

    Returns
    -------
    subthresh_min_amp: float min amplitude
    min_amp_delta: min increment in the stimulus amplitude
    """

    amps_diff = np.round(np.diff(sorted(stim_amps)), decimals=decimals)

    amps_diff = amps_diff[amps_diff > 0]    # remove zeros, repeats are okay
    amp_deltas = np.unique(amps_diff)   # unique nonzero deltas

    if len(amp_deltas) == 0:
        raise IndexError("All stimuli have identical amplitudes")

    min_amp_delta = np.min(amp_deltas)

    if len(amp_deltas) != 1:
        logging.warning(
            "Found multiple coarse long square amplitude step differences: %s.  Using: %f" % (str(amp_deltas), min_amp_delta))

    subthresh_min_amp = SUBTHRESHOLD_LONG_SQUARE_MIN_AMPS.get(min_amp_delta, None)

    if subthresh_min_amp is None:
        raise er.FeatureError("Unknown coarse long square amplitude delta: %f" % min_amp_delta)

    return subthresh_min_amp, min_amp_delta


@record_errors
@fallback_on_error()
def extract_cell_long_square_features(data_set, subthresh_min_amp=None):
    lu.log_pretty_header("Long Squares:", level=2)

    long_square_sweep_numbers = data_set.get_sweep_numbers(
        data_set.ontology.long_square_names,
        clamp_mode=data_set.CURRENT_CLAMP)
    if len(long_square_sweep_numbers) == 0:
        raise er.FeatureError("No long_square sweeps available for feature extraction")

    if subthresh_min_amp is None:
        clsq_sweeps = data_set.filtered_sweep_table(
                        clamp_mode=data_set.CURRENT_CLAMP,
                        stimuli=data_set.ontology.coarse_long_square_names)
        clsq_sweep_numbers = clsq_sweeps['sweep_number'].sort_values().values
        if len(clsq_sweep_numbers) > 0:
            subthresh_min_amp, clsq_amp_delta = select_subthreshold_min_amplitude(clsq_sweeps['stimulus_amplitude'])
            logging.info("Coarse long squares: %f pA step size.  Using subthreshold minimum amplitude of %f.", clsq_amp_delta, subthresh_min_amp)
        else:
            subthresh_min_amp = -100
            logging.info("Assigned subthreshold minimum amplitude of %f.", subthresh_min_amp)

    lsq_sweeps = data_set.sweep_set(long_square_sweep_numbers)
    lsq_sweeps.align_to_start_of_epoch("experiment")

    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(
                                        lsq_sweeps.sweeps[0].i,
                                        lsq_sweeps.sweeps[0].t)

    lsq_spx, lsq_spfx = extractors_for_sweeps(lsq_sweeps,
                                              start=lsq_start,
                                              end=lsq_start+lsq_dur,
                                              **detection_parameters(StimulusType.LONG_SQUARE))

    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
                                    subthresh_min_amp=subthresh_min_amp)

    lsq_features = lsq_an.analyze(lsq_sweeps)

    long_squares_features = lsq_an.as_dict(
        lsq_features,
        [dict(sweep_number=sn) for sn in long_square_sweep_numbers]
        )

    if long_squares_features["hero_sweep"] is None:
        raise er.FeatureError("Could not find hero sweep.")

    return long_squares_features


@record_errors
@fallback_on_error()
def extract_cell_short_square_features(data_set):
    lu.log_pretty_header("Short Squares:", level=2)

    short_square_sweep_numbers = data_set.get_sweep_numbers(
        data_set.ontology.short_square_names,
        clamp_mode=data_set.CURRENT_CLAMP)
    if len(short_square_sweep_numbers) == 0:
        raise er.FeatureError("No short square sweeps available for feature extraction")

    ssq_sweeps = data_set.sweep_set(short_square_sweep_numbers)
    ssq_sweeps.align_to_start_of_epoch("experiment")

    ssq_start, ssq_dur, _, _, _ = stf.get_stim_characteristics(
                                        ssq_sweeps.sweeps[0].i,
                                        ssq_sweeps.sweeps[0].t)

    SSQ_WINDOW = 0.001
    ssq_spx, ssq_spfx = extractors_for_sweeps(ssq_sweeps,
                                              est_window=[ssq_start, ssq_start+SSQ_WINDOW],
                                              **detection_parameters(StimulusType.SHORT_SQUARE))

    ssq_an = spa.ShortSquareAnalysis(ssq_spx, ssq_spfx)

    ssq_features = ssq_an.analyze(ssq_sweeps)

    return ssq_an.as_dict(
        ssq_features,
        [dict(sweep_number=sn) for sn in short_square_sweep_numbers]
        )


@record_errors
@fallback_on_error()
def extract_cell_ramp_features(data_set):
    lu.log_pretty_header("Ramps:", level=2)

    ramp_sweep_numbers = data_set.get_sweep_numbers(
        data_set.ontology.ramp_names,
        clamp_mode=data_set.CURRENT_CLAMP)
    if len(ramp_sweep_numbers) == 0:
        raise er.FeatureError("No ramp sweeps available for feature extraction")

    ramp_sweeps = data_set.sweep_set(ramp_sweep_numbers)
    ramp_sweeps.align_to_start_of_epoch("experiment")

    ramp_start, ramp_dur, _, _, _ = stf.get_stim_characteristics(ramp_sweeps.sweeps[0].i, ramp_sweeps.sweeps[0].t)

    ramp_spx, ramp_spfx = extractors_for_sweeps(ramp_sweeps,
                                                start=ramp_start,
                                                **detection_parameters(StimulusType.RAMP))
    ramp_an = spa.RampAnalysis(ramp_spx, ramp_spfx)
    ramp_features = ramp_an.analyze(ramp_sweeps)
    return ramp_an.as_dict(
        ramp_features,
        [dict(sweep_number=sn) for sn in ramp_sweep_numbers]
        )


def extract_data_set_features(data_set, subthresh_min_amp=None):
    """

    Parameters
    ----------
    data_set : EphysDataSet
        data set
    subthresh_min_amp

    Returns
    -------
    cell_features :

    sweep_features :

    cell_record :

    sweep_records :

    """

    # extract cell-level features
    lu.log_pretty_header("Analyzing cell features:", level=2)

    cell_state = {"failed_fx": False, "fail_fx_message": None}
    feature_states = {}
    cell_features = {}

    (cell_features['long_squares'], feature_states['long_squares_state']) = \
        extract_cell_long_square_features(
            data_set,
            subthresh_min_amp)

    (cell_features['short_squares'], feature_states['short_squares_state']) = \
        extract_cell_short_square_features(data_set)

    (cell_features['ramps'], feature_states['ramps_state']) = \
        extract_cell_ramp_features(data_set)

    # compute sweep features
    iclamp_sweeps = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP)
    sweep_features, feature_states['sweep_features_state'] = \
        extract_sweep_features(data_set, iclamp_sweeps)

    # shuffle peak deflection for the subthreshold long squares
    if not feature_states['long_squares_state']['failed_fx']:
        for s in cell_features["long_squares"]["subthreshold_sweeps"]:
            sweep_features[s['sweep_number']]['peak_deflect'] = s['peak_deflect']

    # if all failed, set cell state to failed
    if all([feature_states[feature]['failed_fx'] for feature
            in feature_states.keys()]):

        cell_state['failed_fx'] = True
        message = '; '.join([feature_states[feature]['fail_fx_message']
                             for feature in feature_states.keys()])
        cell_state['fail_fx_message'] = message

    cell_record = fr.build_cell_feature_record(cell_features)
    sweep_records = fr.build_sweep_feature_record(data_set.sweep_table, sweep_features)

    return (cell_features, sweep_features, cell_record, sweep_records,
            cell_state, feature_states)
