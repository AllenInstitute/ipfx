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
import pytest
import numpy as np
from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor


def test_extractors_no_values():

    SpikeFeatureExtractor()
    SpikeTrainFeatureExtractor(start=0, end=0)


def test_extractor_wrong_inputs(spike_test_pair):
    data = spike_test_pair

    t = data[:, 0]
    v = data[:, 1]
    i = np.zeros_like(v)

    ext = SpikeFeatureExtractor()

    with pytest.raises(IndexError):
        ext.process([t], v, i)

    with pytest.raises(IndexError):
        ext.process([t], [v], i)

    with pytest.raises(ValueError):
        ext.process([t, t], [v], [i])

    with pytest.raises(ValueError):
        ext.process([t, t], [v, v], [i])


def test_extractor_on_sample_data(spike_test_pair):
    data = spike_test_pair

    t = data[:, 0]
    v = data[:, 1]

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t=t, v=v, i=None)

    sx = SpikeTrainFeatureExtractor(start=0, end=t[-1])
    sx.process(t=t, v=v, i=None, spikes_df=spikes)


def test_extractor_on_sample_data_with_i(spike_test_pair):
    data = spike_test_pair

    t = data[:, 0]
    v = data[:, 1]
    i = np.zeros_like(v)

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i)

    sx = SpikeTrainFeatureExtractor(start=0, end=t[-1])
    sx.process(t, v, i, spikes)


def test_extractor_on_zero_voltage():
    t = np.arange(0, 4000) * 2e-5

    v = np.zeros_like(t)
    i = np.zeros_like(t)

    ext = SpikeFeatureExtractor()
    ext.process(t, v, i)


def test_extractor_on_variable_time_step(spike_test_var_dt):
    data = spike_test_var_dt

    t = data[:, 0]
    v = data[:, 1]

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i=None)

    expected_thresh_ind = np.array([73, 183, 314, 463, 616, 770])
    assert np.allclose(spikes["threshold_index"].values, expected_thresh_ind)


def test_extractor_with_high_init_dvdt(spike_test_high_init_dvdt):
    data = spike_test_high_init_dvdt

    t = data[:, 0]
    v = data[:, 1]

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i=None)

    expected_thresh_ind = np.array([11222, 16258, 24060])
    assert np.allclose(spikes["threshold_index"].values, expected_thresh_ind)
