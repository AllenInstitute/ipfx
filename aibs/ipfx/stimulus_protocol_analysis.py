from . import ephys_extractor as efex
import numpy as np
import logging
from collections import Counter


class StimulusProtocolAnalysis(object):
    MEAN_FEATURES = [ "upstroke_downstroke_ratio", "peak_v", "peak_t", "trough_v", "trough_t",
                      "fast_trough_v", "fast_trough_t", "slow_trough_v", "slow_trough_t",
                      "threshold_v", "threshold_i", "threshold_t", "peak_v", "peak_t" ]

    def __init__(self, extractor):
        self.extractors = { 'all': extractor }
        self.features = {}

    @property
    def subthreshold_extractor(self):
        if 'subthreshold' not in self.extractors:
            subthreshold_sweeps = [sweep for sweep in self.extractor.sweeps() if sweep.sweep_feature("avg_rate") == 0]
            self.add_extractor('subthreshold', subthreshold_sweeps)

        return self.extractors['subthreshold']

    @property
    def suprathreshold_extractor(self):
        if 'suprathreshold' not in self.extractors:
            suprathreshold_sweeps = [sweep for sweep in self.extractor.sweeps() if sweep.sweep_feature("avg_rate") > 0]
            self.add_extractor('suprathreshold', suprathreshold_sweeps)
            
        return self.extractors['suprathreshold']

    def add_extractor(self, name, sweeps):
        ext = efex.EphysSweepSetFeatureExtractor.from_sweeps(sweeps)
        self.extractors[name] = ext
        return ext

    def mean_features_first_spike(self, sweeps, features_list=None):
        """ Compute mean feature values for the first spike in list of extractors """

        if features_list is None:
            features_list = self.MEAN_FEATURES

        output = {}
        for mf in features_list:
            mfd = [ sweep.spikes()[0][mf] for sweep in sweeps if sweep.sweep_feature("avg_rate") > 0 ]
            output[mf] = np.mean(mfd)
        return output

    @property
    def extractor(self):
        return self.extractors['all']

    def analyze(self):
        pass

    def as_dict(self):
        return {}

class RampAnalysis(StimulusProtocolAnalysis):
    def analyze(self): 
        self.extractor.process_spikes()
        spiking_ext = self.suprathreshold_extractor
        self.features["spiking_sweeps"] = spiking_ext.sweeps()
        self.features["mean_spike_0"] = self.mean_features_first_spike(spiking_ext.sweeps())

    def as_dict(self):
        out = self.features.copy()
        out["spiking_sweeps"] = [ s.as_dict() for s in out["spiking_sweeps"] ]
        return out

class LongSquareAnalysis(StimulusProtocolAnalysis):
    SUBTHRESH_MAX_AMP = 0
    SAG_TARGET = -100.
    HERO_MIN_AMP_OFFSET = 39.0
    HERO_MAX_AMP_OFFSET = 61.0

    def __init__(self, extractor, subthresh_min_amp):
        super(LongSquareAnalysis, self).__init__(extractor)
        self.subthresh_min_amp = subthresh_min_amp

    def analyze(self): 
        ext = self.extractor
        ext.process_spikes()

        self.features["sweeps"] = ext.sweeps()
        for s in ext.sweeps():
            s.set_stimulus_amplitude_calculator(efex._step_stim_amp)

        self.analyze_suprathreshold()
        self.analyze_subthreshold()


    def analyze_suprathreshold(self):
        ext = self.extractor
        spiking_indexes = np.flatnonzero(ext.sweep_features("avg_rate"))

        if len(spiking_indexes) == 0:
            raise ft.FeatureError("No spiking long square sweeps, cannot compute cell features.")

        amps = ext.sweep_features("stim_amp")#self.long_squares_stim_amps()
        min_index = np.argmin(amps[spiking_indexes])
        rheobase_index = spiking_indexes[min_index]
        rheobase_i = efex._step_stim_amp(ext.sweeps()[rheobase_index])

        self.features["rheobase_extractor_index"] = rheobase_index
        self.features["rheobase_i"] = rheobase_i
        self.features["rheobase_sweep"] = ext.sweeps()[rheobase_index]

        spiking_ext = self.suprathreshold_extractor
        self.features["spiking_sweeps"] = spiking_ext.sweeps()
        self.features["fi_fit_slope"] = efex.fit_fi_slope(spiking_ext)

        # find hero sweep
        hero_sweep = self.find_hero_sweep(rheobase_i, self.suprathreshold_extractor.sweeps())
        self.features['hero_sweep'] = hero_sweep

    def analyze_subthreshold(self):
        subthresh_ext = self.subthreshold_extractor
        subthresh_sweeps = subthresh_ext.sweeps()

        if len(subthresh_sweeps) == 0:
            raise ft.FeatureError("No subthreshold long square sweeps, cannot evaluate cell features.")

        peaks = subthresh_ext.sweep_features("peak_deflect")
        sags = subthresh_ext.sweep_features("sag")
        sag_eval_levels = np.array([sweep.voltage_deflection()[0] for sweep in subthresh_ext.sweeps()])
        target_level = self.SAG_TARGET
        closest_index = np.argmin(np.abs(sag_eval_levels - target_level))

        self.features["sag"] = sags[closest_index]
        self.features["vm_for_sag"] = sag_eval_levels[closest_index]
        self.features["subthreshold_sweeps"] = subthresh_ext.sweeps()
        for s in self.features["subthreshold_sweeps"]:
            s.set_stimulus_amplitude_calculator(efex._step_stim_amp)

        logging.debug("subthresh_sweeps: %d", len(subthresh_sweeps))
        calc_subthresh_sweeps = [sweep for sweep in subthresh_sweeps if
                                 sweep.sweep_feature("stim_amp") < self.SUBTHRESH_MAX_AMP and
                                 sweep.sweep_feature("stim_amp") > self.subthresh_min_amp]
        calc_subthresh_ext = self.add_extractor('subthreshold_membrane_property', calc_subthresh_sweeps)

        logging.debug("calc_subthresh_sweeps: %d", len(calc_subthresh_sweeps))        

        self.features["subthreshold_membrane_property_sweeps"] = calc_subthresh_ext.sweeps()
        self.features["input_resistance"] = efex.input_resistance(calc_subthresh_ext)
        self.features["tau"] = efex.membrane_time_constant(calc_subthresh_ext)
        self.features["v_baseline"] = np.nanmean(self.extractor.sweep_features("v_baseline"))


    def as_dict(self):
        out = self.features.copy()
        
        out["sweeps"] = [ s.as_dict() for s in out["sweeps"] ]
        out["spiking_sweeps"] = [ s.as_dict() for s in out["spiking_sweeps"] ]
        out["subthreshold_sweeps"] = [ s.as_dict() for s in out["subthreshold_sweeps"] ]
        out["subthreshold_membrane_property_sweeps"] = [ s.as_dict() for s in out["subthreshold_membrane_property_sweeps"] ]
        out["rheobase_sweep"] = out["rheobase_sweep"].as_dict()
        out["hero_sweep"] = out["hero_sweep"].as_dict() if out.get("hero_sweep", None) else None

        return out

    def find_hero_sweep(self, rheo_amp, long_square_spiking_sweeps, 
                        min_offset=HERO_MIN_AMP_OFFSET, 
                        max_offset=HERO_MAX_AMP_OFFSET):

        hero_min, hero_max = rheo_amp + min_offset, rheo_amp + max_offset
        hero_amp = float("inf")
        hero_sweep = None

        for sweep in long_square_spiking_sweeps:
            nspikes = len(sweep.spikes())
            amp = sweep.sweep_feature("stim_amp")

            if nspikes > 0 and amp > hero_min and amp < hero_max and amp < hero_amp:
                hero_amp = amp
                hero_sweep = sweep

        return hero_sweep



class ShortSquareAnalysis(StimulusProtocolAnalysis): 
    def analyze(self):
        self.extractor.process_spikes()

        spiking_ext = self.suprathreshold_extractor
        spiking_sweeps = spiking_ext.sweeps()

        # Need to count how many had spikes at each amplitude; find most; ties go to lower amplitude
        if len(spiking_sweeps) == 0:
            raise ft.FeatureError("No spiking short square sweeps, cannot compute cell features.")

        most_common = Counter(map(efex._short_step_stim_amp, spiking_sweeps)).most_common()
        common_amp, common_count = most_common[0]
        for c in most_common[1:]:
            if c[1] < common_count:
                break
            if c[0] < common_amp:
                common_amp = c[0]

        ca_sweeps = [sweep for sweep in spiking_sweeps if efex._short_step_stim_amp(sweep) == common_amp]
        ca_ext = self.add_extractor('common_amp', ca_sweeps)

        self.features["stimulus_amplitude"] = common_amp
        self.features["common_amp_sweeps"] = ca_ext.sweeps()

        for s in ca_ext.sweeps():
            s.set_stimulus_amplitude_calculator(efex._short_step_stim_amp)

        self.features["mean_spike_0"] = self.mean_features_first_spike(ca_ext.sweeps())

            
    def as_dict(self):
        out = self.features.copy()
        out["common_amp_sweeps"] = [ s.as_dict() for s in out["common_amp_sweeps"] ]
        return out
