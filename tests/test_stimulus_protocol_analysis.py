import pytest
import pandas as pd
import numpy as np
from ipfx.stimulus_protocol_analysis import LongSquareAnalysis
from ipfx.feature_extractor import SpikeTrainFeatureExtractor, SpikeFeatureExtractor


@pytest.fixture()
def spiking_sweep_features():
    df = pd.DataFrame({"avg_rate": np.array([1, 3, 7, 8, 14]),
                       "stim_amp": np.array([-10, 20, 40, 50, 60])}, index=[4, 5, 7, 8, 9])

    return df


@pytest.fixture()
def long_square_analysis():
    # build the extractors
    spx = SpikeFeatureExtractor(start=0.27, end=1.27)
    spfx = SpikeTrainFeatureExtractor(start=0.27, end=1.27)
    return LongSquareAnalysis(spx, spfx, subthresh_min_amp=-100.0)


def test_find_rheobase_sweep(long_square_analysis, spiking_sweep_features):

    spiking_features_with_negative_amplitude = spiking_sweep_features
    rheobase_sweep = long_square_analysis.find_rheobase_sweep(spiking_features_with_negative_amplitude)
    assert rheobase_sweep["stim_amp"] == 20


def test_find_hero_sweep(long_square_analysis, spiking_sweep_features):

    spiking_features_with_negative_amplitude = spiking_sweep_features
    rheobase_sweep = long_square_analysis.find_rheobase_sweep(spiking_features_with_negative_amplitude)
    rheobase_i = rheobase_sweep["stim_amp"]

    hero_sweep = long_square_analysis.find_hero_sweep(rheobase_i, spiking_sweep_features)
    assert hero_sweep["stim_amp"] == 60
