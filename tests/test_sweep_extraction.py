from __future__ import print_function
import numpy as np
from ipfx.feature_extractor import SpikeFeatureExtractor


def test_extractor_on_variable_time_step(spike_test_var_dt):
    data = spike_test_var_dt

    print(data)
    t = data[:, 0]
    v = data[:, 1]

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i=None)

    expected_thresh_ind = np.array([73, 183, 314, 463, 616, 770])
    assert np.allclose(spikes["threshold_index"].values, expected_thresh_ind)
