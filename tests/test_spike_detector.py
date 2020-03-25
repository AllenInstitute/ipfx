import pytest
import ipfx.spike_detector as spkd
import numpy as np
import ipfx.error as er


def test_v_and_t_are_arrays():
    v = [0, 1, 2]
    t = [0, 1, 2]
    with pytest.raises(TypeError):
        spkd.detect_putative_spikes(v, t)

    with pytest.raises(TypeError):
        spkd.detect_putative_spikes(np.array(v), t)


def test_size_mismatch():
    v = np.array([0, 1, 2])
    t = np.array([0, 1])
    with pytest.raises(er.FeatureError):
        spkd.detect_putative_spikes(v, t)


def test_detect_one_spike(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    expected_spikes = np.array([728])

    assert np.allclose(spkd.detect_putative_spikes(v[:3000], t[:3000]), expected_spikes)


def test_detect_two_spikes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    expected_spikes = np.array([728, 3386])

    assert np.allclose(spkd.detect_putative_spikes(v, t), expected_spikes)


def test_detect_no_spikes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = np.zeros_like(t)

    assert len(spkd.detect_putative_spikes(v, t)) == 0


def test_detect_no_spike_peaks(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = np.zeros_like(t)
    spikes = np.array([])

    assert len(spkd.find_peak_indexes(v, t, spikes)) == 0


def test_detect_two_spike_peaks(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    expected_peaks = np.array([812, 3478])

    assert np.allclose(spkd.find_peak_indexes(v, t, spikes), expected_peaks)


def test_filter_problem_spikes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    peaks = np.array([812, 3478])

    new_spikes, new_peaks = spkd.filter_putative_spikes(v, t, spikes, peaks)
    assert np.allclose(spikes, new_spikes)
    assert np.allclose(peaks, new_peaks)


def test_filter_no_spikes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = np.zeros_like(t)
    spikes = np.array([])
    peaks = np.array([])

    new_spikes, new_peaks = spkd.filter_putative_spikes(v, t, spikes, peaks)
    assert len(new_spikes) == len(new_peaks) == 0


def test_find_clipped_spikes(spike_test_pair):

    data = spike_test_pair
    spike_indexes = np.array([725, 3382])
    peak_indexes = np.array([812, 3478])
    t = data[:, 0]
    v = data[:, 1]

    clipped = spkd.find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index=3550, tol=1)
    print((clipped, np.array([False, True])))
    print((clipped.dtype, np.array([False, True]).dtype))
    assert np.array_equal(clipped, [False, True])  # last spike is clipped

    clipped = spkd.find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index=3600, tol=1)
    print((clipped, np.array([False, False])))
    assert np.array_equal(clipped, [False, False])  # last spike is Ok

    spike_indexes = np.array([])
    peak_indexes = np.array([])

    t = data[1500:3000, 0]
    v = data[1500:3000, 1]

    clipped = spkd.find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index=3600, tol=1)
    assert np.array_equal(clipped, [])  # no spikes in the trace


def test_downstrokes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741])

    expected_downstrokes = np.array([862, 3532])
    assert np.allclose(spkd.find_downstroke_indexes(v, t, peaks, troughs), expected_downstrokes)


def test_downstrokes_too_many_troughs(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    peaks = np.array([812, 3478])
    troughs = np.array([1089, 3741, 3999])

    with pytest.raises(er.FeatureError):
        spkd.find_downstroke_indexes(v, t, peaks, troughs)


def test_troughs(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])

    expected_troughs = np.array([1089, 3741])
    assert np.allclose(spkd.find_trough_indexes(v, t, spikes, peaks), expected_troughs)


def test_troughs_with_peak_at_end(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([725, 3382])
    peaks = np.array([812, 3478])
    clipped = np.array([False, True])

    troughs = spkd.find_trough_indexes(v[:peaks[-1]], t[:peaks[-1]],
                                       spikes, peaks, clipped=clipped)
    assert np.isnan(troughs[-1])


def test_check_spikes_and_peaks():
    t = np.arange(0, 30) * 5e-6
    v = np.zeros_like(t)
    spikes = np.array([0, 5])
    peaks = np.array([10, 15])
    upstrokes = np.array([3, 13])

    new_spikes, new_peaks, new_upstrokes, clipped = spkd.check_thresholds_and_peaks(v, t, spikes, peaks, upstrokes)
    assert np.allclose(new_spikes, spikes[:-1])
    assert np.allclose(new_peaks, peaks[1:])


def test_thresholds(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    upstrokes = np.array([778, 3440])

    expected_thresholds = np.array([725, 3382])
    assert np.allclose(spkd.refine_threshold_indexes(v, t, upstrokes), expected_thresholds)


def test_thresholds_cannot_find_target(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    upstrokes = np.array([778, 3440])

    expected = np.array([0, 778])
    assert np.allclose(spkd.refine_threshold_indexes(v, t, upstrokes, thresh_frac=-5.0), expected)


def test_upstrokes(spike_test_pair):
    data = spike_test_pair
    t = data[:, 0]
    v = data[:, 1]
    spikes = np.array([728, 3386])
    peaks = np.array([812, 3478])

    expected_upstrokes = np.array([778, 3440])
    assert np.allclose(spkd.find_upstroke_indexes(v, t, spikes, peaks), expected_upstrokes)
