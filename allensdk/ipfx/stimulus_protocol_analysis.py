import numpy as np
import pandas as pd
import logging
from collections import Counter

from . import ephys_features as ft
from . import ephys_extractor as efex

from .ephys_data_set import SweepSet

class StimulusProtocolAnalysis(object):
    MEAN_FEATURES = [ "upstroke_downstroke_ratio", "peak_v", "peak_t", "trough_v", "trough_t",
                      "fast_trough_v", "fast_trough_t", "slow_trough_v", "slow_trough_t",
                      "threshold_v", "threshold_i", "threshold_t", "peak_v", "peak_t" ]

    def __init__(self, spx, sptx):
        self.spx = spx
        self.sptx = sptx

        self._spikes_set = None
        self._sweep_features = None

    def _sweep_to_dict(self, sweep, extra_params=None):
        s = sweep.to_dict()
        s['index'] = sweep.name
        s['spikes'] = self._spikes_set[sweep.name].to_dict(orient='records')
        if extra_params:
            s.update(extra_params[s['index']])
        return s

    def _sweeps_to_dict(self, sweeps, extra_params=None):
        sweep_index = sweeps.to_dict(orient="index")
        out = []
        for sid in sweeps.index:
            s = sweep_index[sid]
            s['index'] = sid
            s['spikes'] = self._spikes_set[sid].to_dict(orient='records')
            if extra_params:
                s.update(extra_params[sid])
            out.append(s)
        return out

    def subthreshold_sweeps(self, sweep_features=None):
        if sweep_features is None:
            sweep_features = self.all_sweep_features()

        return sweep_features["avg_rate"] == 0

    def suprathreshold_sweeps(self, sweep_features=None):
        if sweep_features is None:
            sweep_features = self.all_sweep_features()

        return sweep_features["avg_rate"] > 0

    def all_sweep_features(self):
        return self._sweep_features

    def mean_features_first_spike(self, spikes_set, features_list=None):
        """ Compute mean feature values for the first spike in list of extractors """

        if features_list is None:
            features_list = self.MEAN_FEATURES

        output = {}
        for mf in features_list:
            mfd = [ spikes[mf].values[0] for spikes in spikes_set if len(spikes) > 0 ]
            output[mf] = np.mean(mfd)
        return output

    def analyze_basic_features(self, sweep_set, extra_sweep_features=None):
        self._spikes_set = [ self.spx.process(sweep.t, sweep.v, sweep.i)
                             for sweep in sweep_set.sweeps ]

        self._sweep_features = pd.DataFrame([ self.sptx.process(sweep.t, sweep.v, sweep.i, spikes, extra_sweep_features)
                                              for sweep, spikes in zip(sweep_set.sweeps, self._spikes_set) ])

#        print "self._sweep_features", self._sweep_features

    def reset_basic_features(self):
        self._spikes_set = None
        self._sweep_features = None

    def analyze(self, sweep_set, extra_sweep_features=None):
        return {}

    def as_dict(self, features, extra_params=None):
        return {}

class RampAnalysis(StimulusProtocolAnalysis):
    def analyze(self, sweep_set):
        self.analyze_basic_features(sweep_set)

        features = {}

        spiking_sweeps = self.suprathreshold_sweeps()
        features["spiking_sweeps"] = self._sweep_features[spiking_sweeps]
        features["mean_spike_0"] = self.mean_features_first_spike(self._spikes_set)

        return features

    def as_dict(self, features, extra_params=None):
        out = features.copy()
        for k in [ "spiking_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)

        return out


class LongSquareAnalysis(StimulusProtocolAnalysis):

    SUBTHRESH_MAX_AMP = 0
    SAG_TARGET = -100.
    HERO_MIN_AMP_OFFSET = 39.0
    HERO_MAX_AMP_OFFSET = 61.0

    def __init__(self, spx, sptx, subthresh_min_amp, tau_frac=0.1):
        super(LongSquareAnalysis, self).__init__(spx, sptx)
        self.subthresh_min_amp = subthresh_min_amp
        self.sptx.stim_amp_fn = ft._step_stim_amp
        self.tau_frac = tau_frac

    def analyze(self, sweep_set):

        extra_sweep_features = ['peak_deflect','stim_amp','v_baseline','sag','burst','delay']
        self.analyze_basic_features(sweep_set, extra_sweep_features=extra_sweep_features)

        features = {}

        features["sweeps"] = self._sweep_features
        features["v_baseline"] = np.nanmean(self._sweep_features["v_baseline"].values)

        spiking_features = self.analyze_suprathreshold(sweep_set)
        subthresh_features = self.analyze_subthreshold(sweep_set)

        features.update(spiking_features)
        features.update(subthresh_features)

        return features

    def analyze_suprathreshold(self, sweep_set):
        features = {}
        spiking_sweeps = self.suprathreshold_sweeps()

        if len(spiking_sweeps) == 0:
            raise ft.FeatureError("No spiking long square sweeps, cannot compute cell features.")

        spiking_features = self._sweep_features[spiking_sweeps]

        min_index = np.argmin(spiking_features["stim_amp"].values)

        rheobase_index = spiking_features.iloc[min_index].name

        rheo_sweep = sweep_set.sweeps[rheobase_index]
        rheobase_i = ft._step_stim_amp(rheo_sweep.t, rheo_sweep.i, self.spx.start)

        features["rheobase_extractor_index"] = rheobase_index
        features["rheobase_i"] = rheobase_i
        features["rheobase_sweep"] = spiking_features.iloc[min_index]
        features["spiking_sweeps"] = spiking_features

        features["fi_fit_slope"] = efex.fit_fi_slope(spiking_features["stim_amp"].values,
                                                     spiking_features["avg_rate"].values)

        # find hero sweep
        hero_sweep = self.find_hero_sweep(rheobase_i, spiking_features)

        features['hero_sweep'] = hero_sweep

        return features

    def analyze_subthreshold(self, sweep_set):
        features = {}

        subthresh_sweeps = self.subthreshold_sweeps()
        subthresh_features = self._sweep_features[subthresh_sweeps]

        if len(subthresh_features) == 0:
            raise ft.FeatureError("No subthreshold long square sweeps, cannot evaluate cell features.")

        logging.debug("subthresh_sweeps: %d", len(subthresh_features))

        sags = subthresh_features["sag"]
        sag_eval_levels = np.array([ v[0] for v in subthresh_features["peak_deflect"] ])
        closest_index = np.argmin(np.abs(sag_eval_levels - self.SAG_TARGET))

        features["sag"] = sags.values[closest_index]
        features["vm_for_sag"] = sag_eval_levels[closest_index]
        features["subthreshold_sweeps"] = subthresh_features

        calc_subthresh_features = subthresh_features[ (subthresh_features["stim_amp"] < self.SUBTHRESH_MAX_AMP) & \
                                                      (subthresh_features["stim_amp"] > self.subthresh_min_amp) ].copy()

        logging.debug("calc_subthresh_sweeps: %d", len(calc_subthresh_features))

        calc_subthresh_ss = SweepSet([sweep_set.sweeps[i] for i in calc_subthresh_features.index.values])
        taus = [ ft.time_constant(s.t, s.v, s.i, self.spx.start, self.spx.end, self.tau_frac, self.sptx.baseline_interval) for s in calc_subthresh_ss.sweeps ]
        calc_subthresh_features['tau'] = taus

        features["subthreshold_membrane_property_sweeps"] = calc_subthresh_features
        features["input_resistance"] = ft.input_resistance(calc_subthresh_ss.t,
                                                           calc_subthresh_ss.i,
                                                           calc_subthresh_ss.v,
                                                           self.spx.start, self.spx.end,
                                                           self.sptx.baseline_interval)

        features["tau"] = np.nanmean(calc_subthresh_features['tau'])

        return features

    def as_dict(self, features, extra_params=None):
        out = features.copy()

        for k in [ "sweeps", "subthreshold_membrane_property_sweeps", "subthreshold_sweeps", "spiking_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)

        for k in [ "hero_sweep", "rheobase_sweep" ]:
            out[k] = self._sweep_to_dict(out[k], extra_params)

        return out

    def find_hero_sweep(self, rheo_amp, spiking_features,
                        min_offset=HERO_MIN_AMP_OFFSET,
                        max_offset=HERO_MAX_AMP_OFFSET):

        hero_min, hero_max = rheo_amp + min_offset, rheo_amp + max_offset
        spiking_features = spiking_features.sort_values("stim_amp")
        hero_features = spiking_features[(spiking_features["stim_amp"] > hero_min) & (spiking_features["stim_amp"] < hero_max)]

        if len(hero_features) == 0:
#            return None
            raise ValueError
        else:
            return hero_features.iloc[0]


class ShortSquareAnalysis(StimulusProtocolAnalysis):
    def __init__(self, spx, sptx):
        super(ShortSquareAnalysis, self).__init__(spx, sptx)
        self.sptx.stim_amp_fn = ft._short_step_stim_amp

    def analyze(self, sweep_set):
        extra_sweep_features = [ "stim_amp" ]
        self.analyze_basic_features(sweep_set, extra_sweep_features=extra_sweep_features)

        features = {}

        spiking_sweeps = self.suprathreshold_sweeps()
        spiking_features = self._sweep_features[spiking_sweeps]

        # Need to count how many had spikes at each amplitude; find most; ties go to lower amplitude
        if len(spiking_sweeps) == 0:
            raise ft.FeatureError("No spiking short square sweeps, cannot compute cell features.")

        most_common = Counter(spiking_features["stim_amp"].values).most_common()
        common_amp, common_count = most_common[0]
        for c in most_common[1:]:
            if c[1] < common_count:
                break
            if c[0] < common_amp:
                common_amp = c[0]

        ca_features = spiking_features[spiking_features["stim_amp"] == common_amp]
        features["stimulus_amplitude"] = common_amp
        features["common_amp_sweeps"] = ca_features
        features["mean_spike_0"] = self.mean_features_first_spike(self._spikes_set)

        return features


    def as_dict(self, features, extra_params=None):
        out = features.copy()
        for k in [ "common_amp_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)
        return out
