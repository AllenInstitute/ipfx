import numpy as np
import pandas as pd
import logging
from collections import Counter
from . import stim_features as stf
from . import subthresh_features as subf
from . import spike_train_features as strf
from .sweep import SweepSet
from . import error as er


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

    def subthreshold_sweep_features(self, sweep_features=None):
        if sweep_features is None:
            sweep_features = self.all_sweep_features()

        return sweep_features[sweep_features["avg_rate"] == 0]

    def suprathreshold_sweep_features(self, sweep_features=None):
        if sweep_features is None:
            sweep_features = self.all_sweep_features()

        return sweep_features[sweep_features["avg_rate"] > 0]

    def all_sweep_features(self):
        return self._sweep_features

    def mean_features_first_spike(self, spikes_set, features_list=None):
        """ Compute mean feature values for the first spike in list of extractors """

        if features_list is None:
            features_list = self.MEAN_FEATURES

        output = {}
        for mf in features_list:
            mfd = [ spikes[mf].values[0] for spikes in spikes_set if len(spikes) > 0 ]
            output[mf] = np.nanmean(mfd)
        return output

    def analyze_basic_features(self, sweep_set, extra_sweep_features=None, exclude_clipped=False):
        self._spikes_set = []
        for sweep in sweep_set.sweeps:
            self._spikes_set.append(self.spx.process(sweep.t, sweep.v, sweep.i))

        self._sweep_features = pd.DataFrame([ self.sptx.process(sweep.t, sweep.v, sweep.i, spikes, extra_sweep_features, exclude_clipped=exclude_clipped)
                                              for sweep, spikes in zip(sweep_set.sweeps, self._spikes_set) ])

    def reset_basic_features(self):
        self._spikes_set = None
        self._sweep_features = None

    def analyze(self, sweep_set, extra_sweep_features=None, exclude_clipped=False):
        self.analyze_basic_features(sweep_set, extra_sweep_features=extra_sweep_features, exclude_clipped=exclude_clipped)
        return {"spikes_set": self._spikes_set, "sweeps": self._sweep_features}

    def as_dict(self, features, extra_params=None):
        return {}


class RampAnalysis(StimulusProtocolAnalysis):
    def analyze(self, sweep_set):
        features = super(RampAnalysis, self).analyze(sweep_set)

        spiking_sweep_features = self.suprathreshold_sweep_features()
        features["spiking_sweeps"] = spiking_sweep_features
        features["mean_spike_0"] = self.mean_features_first_spike(self._spikes_set)

        return features

    def as_dict(self, features, extra_params=None):
        out = features.copy()
        del out["sweeps"]
        del out["spikes_set"]

        for k in [ "spiking_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)

        return out


class LongSquareAnalysis(StimulusProtocolAnalysis):

    SUBTHRESH_MAX_AMP = 0
    SAG_TARGET = -100.
    HERO_MIN_AMP_OFFSET = 39.0
    HERO_MAX_AMP_OFFSET = 61.0

    def __init__(self, spx, sptx, subthresh_min_amp, tau_frac=0.1,
                 require_subthreshold=True, require_suprathreshold=True):
        super(LongSquareAnalysis, self).__init__(spx, sptx)
        self.subthresh_min_amp = subthresh_min_amp
        self.sptx.stim_amp_fn = stf._step_stim_amp
        self.tau_frac = tau_frac
        self.require_subthreshold = require_subthreshold
        self.require_suprathreshold = require_suprathreshold

    def analyze(self, sweep_set):

        extra_sweep_feature_names = ['peak_deflect','stim_amp','v_baseline','sag']
        features = super(LongSquareAnalysis, self).analyze(sweep_set, extra_sweep_features=extra_sweep_feature_names)

        features["v_baseline"] = np.nanmean(self._sweep_features["v_baseline"].values)

        spiking_features = self.analyze_suprathreshold(sweep_set)
        subthresh_features = self.analyze_subthreshold(sweep_set)

        features.update(spiking_features)
        features.update(subthresh_features)

        return features

    def analyze_suprathreshold(self, sweep_set):
        features = {}
        spiking_sweep_features = self.suprathreshold_sweep_features()

        if len(spiking_sweep_features) == 0:
            if self.require_suprathreshold:
                raise er.FeatureError("No spiking long square sweeps, cannot compute cell features.")
            else:
                logging.info("No spiking long square sweeps: cannot compute related cell features.")
                return features

        rheobase_sweep_features = self.find_rheobase_sweep(spiking_sweep_features)

        rheobase_i = rheobase_sweep_features["stim_amp"]

        features["rheobase_i"] = rheobase_i
        features["rheobase_sweep"] = rheobase_sweep_features
        features["spiking_sweeps"] = spiking_sweep_features

        features["fi_fit_slope"] = strf.fit_fi_slope(spiking_sweep_features["stim_amp"].values,
                                                     spiking_sweep_features["avg_rate"].values)

        # find hero sweep
        hero_sweep_features = self.find_hero_sweep(rheobase_i, spiking_sweep_features)

        features['hero_sweep'] = hero_sweep_features

        return features

    def analyze_subthreshold(self, sweep_set):
        features = {}

        subthreshold_sweep_features = self.subthreshold_sweep_features()

        if len(subthreshold_sweep_features) == 0:
            if self.require_subthreshold:
                raise er.FeatureError("No subthreshold long square sweeps, cannot evaluate cell features.")
            else:
                logging.info("No subthreshold long square sweeps: cannot compute related cell features.")
                return features

        sags = subthreshold_sweep_features["sag"]
        sag_eval_levels = np.array([ v[0] for v in subthreshold_sweep_features["peak_deflect"] ])
        closest_index = np.argmin(np.abs(sag_eval_levels - self.SAG_TARGET))

        features["sag"] = sags.values[closest_index]
        features["vm_for_sag"] = sag_eval_levels[closest_index]
        features["subthreshold_sweeps"] = subthreshold_sweep_features

        calc_subthresh_features = subthreshold_sweep_features[ (subthreshold_sweep_features["stim_amp"] < self.SUBTHRESH_MAX_AMP) & \
                                                            (subthreshold_sweep_features["stim_amp"] > self.subthresh_min_amp) ].copy()

        if len(calc_subthresh_features) == 0:
            error_string = F"No subthreshold long square sweeps with stim_amp " \
                           F"in range [{self.subthresh_min_amp,self.SUBTHRESH_MAX_AMP}] " \
                           F"Cannot evaluate cell features."
            if self.require_subthreshold:
                raise er.FeatureError(error_string)
            else:
                return features


        calc_subthresh_ss = SweepSet([sweep_set.sweeps[i] for i in calc_subthresh_features.index.values])
        median_peak_time = np.median([s.t[subf.voltage_deflection(s.t, s.v, s.i, self.spx.start, self.spx.end, "min")[1]]
                                      for s in calc_subthresh_ss.sweeps])
        taus = [ subf.time_constant(s.t, s.v, s.i, self.spx.start, self.spx.end, median_peak_time, self.tau_frac, self.sptx.baseline_interval) for s in calc_subthresh_ss.sweeps ]

        calc_subthresh_features['tau'] = taus

        features["subthreshold_membrane_property_sweeps"] = calc_subthresh_features
        features["input_resistance"] = subf.input_resistance(calc_subthresh_ss.t,
                                                           calc_subthresh_ss.i,
                                                           calc_subthresh_ss.v,
                                                           self.spx.start, self.spx.end,
                                                           self.sptx.baseline_interval)

        features["tau"] = np.nanmean(calc_subthresh_features['tau'])

        return features

    def as_dict(self, features, extra_params=None):
        out = features.copy()
        del out["spikes_set"]

        for k in [ "sweeps", "subthreshold_membrane_property_sweeps", "subthreshold_sweeps", "spiking_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)

        for k in [ "hero_sweep", "rheobase_sweep" ]:
            out[k] = self._sweep_to_dict(out[k], extra_params)

        return out

    def find_rheobase_sweep(self,spiking_features):

        spiking_features = spiking_features.sort_values("stim_amp")

        spiking_features_depolarized = spiking_features[spiking_features["stim_amp"] > 0]

        if spiking_features_depolarized.empty:
            raise ValueError("Cannot find rheobase sweep in spiking sweeps with amplitudes:")
        else:
            return spiking_features_depolarized.iloc[0]

    def find_hero_sweep(self, rheo_amp, spiking_features,
                        min_offset=HERO_MIN_AMP_OFFSET,
                        max_offset=HERO_MAX_AMP_OFFSET):

        hero_min, hero_max = rheo_amp + min_offset, rheo_amp + max_offset
        spiking_features = spiking_features.sort_values("stim_amp")
        sweep_features_range = spiking_features[(spiking_features["stim_amp"] > hero_min) & (spiking_features["stim_amp"] < hero_max)]

        if not sweep_features_range.empty:
            hero_features = sweep_features_range.iloc[0]
            logging.debug("Found hero sweep with amp %f in the range of stim amplitudes: [%f,%f] pA, rheobase amp: %f" % (hero_features["stim_amp"], hero_min, hero_max,rheo_amp))
        else:
            logging.debug("Cannot find hero sweep in the range of stim amplitudes: [%f,%f] pA, rheobase amp: %f" % (hero_min, hero_max,rheo_amp))
            index_hero = abs(hero_min - spiking_features["stim_amp"]).idxmin()
            hero_features = spiking_features.loc[index_hero]
            logging.debug("Selecting as hero sweep with the amplitude %f closest to the min amplitude in [%f,%f] pA " % (hero_features["stim_amp"], hero_min, hero_max))

        if hero_features.empty:
            raise ValueError("Cannot find hero sweep.")

        return hero_features


class ShortSquareAnalysis(StimulusProtocolAnalysis):
    def __init__(self, spx, sptx):
        super(ShortSquareAnalysis, self).__init__(spx, sptx)
        self.sptx.stim_amp_fn = stf._short_step_stim_amp

    def analyze(self, sweep_set):
        extra_sweep_features = [ "stim_amp" ]
        features = super(ShortSquareAnalysis, self).analyze(sweep_set, extra_sweep_features=extra_sweep_features, exclude_clipped=True)

        spiking_sweep_features = self.suprathreshold_sweep_features()

        # Need to count how many had spikes at each amplitude; find most; ties go to lower amplitude
        if len(spiking_sweep_features) == 0:
            raise er.FeatureError("No spiking short square sweeps, cannot compute cell features.")

        most_common = Counter(spiking_sweep_features["stim_amp"].values).most_common()
        common_amp, common_count = most_common[0]

        for c in most_common[1:]:
            if c[1] < common_count:
                break
            if c[0] < common_amp:
                common_amp = c[0]

        ca_features = spiking_sweep_features[spiking_sweep_features["stim_amp"] == common_amp]
        features["stimulus_amplitude"] = common_amp
        features["common_amp_sweeps"] = ca_features
        features["mean_spike_0"] = self.mean_features_first_spike(self._spikes_set)

        return features

    def as_dict(self, features, extra_params=None):
        out = features.copy()
        del out["sweeps"]
        del out["spikes_set"]
        for k in [ "common_amp_sweeps" ]:
            out[k] = self._sweeps_to_dict(out[k], extra_params)
        return out
