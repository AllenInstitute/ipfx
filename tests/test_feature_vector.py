from __future__ import print_function
import numpy as np
import pandas as pd
from ipfx.aibs_data_set import AibsDataSet
import ipfx.data_set_features as dsf
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.feature_vectors as fv
from ipfx.stimulus import StimulusOntology
from ipfx.sweep import Sweep, SweepSet
import allensdk.core.json_utilities as ju
import pytest
import os
from .helpers_for_tests import download_file


TEST_OUTPUT_DIR = "/allen/aibs/informatics/module_test_data/ipfx/test_feature_vector"

ontology = StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))

@pytest.fixture
def feature_vector_input():

    TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

    nwb_file_name = "Pvalb-IRES-Cre;Ai14-415796.02.01.01.nwb"
    nwb_file_full_path = os.path.join(TEST_DATA_PATH, nwb_file_name)

    if not os.path.exists(nwb_file_full_path):
        download_file(nwb_file_name, nwb_file_full_path)

    data_set = AibsDataSet(nwb_file=nwb_file_full_path, ontology=ontology)

    lsq_sweep_numbers = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
                                                      stimuli=ontology.long_square_names).sweep_number.sort_values().values

    lsq_sweeps = data_set.sweep_set(lsq_sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(lsq_sweeps.sweeps[0].i,
                                                               lsq_sweeps.sweeps[0].t)

    lsq_end = lsq_start + lsq_dur
    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                  start=lsq_start,
                                                  end=lsq_end,
                                                  **dsf.detection_parameters(data_set.LONG_SQUARE))
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)

    lsq_features = lsq_an.analyze(lsq_sweeps)

    return lsq_sweeps, lsq_features, lsq_start, lsq_end

@pytest.mark.requires_inhouse_data
def test_isi_shape(feature_vector_input):

    sweeps, features, start, end = feature_vector_input

    isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
        sweeps, features, end - start)
    temp_data = fv.isi_shape(isi_sweep, isi_sweep_spike_info, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "isi_shape.npy"))

    assert np.array_equal(test_data, temp_data)


@pytest.mark.requires_inhouse_data
def test_step_subthreshold(feature_vector_input):
    sweeps, features, start, end = feature_vector_input
    target_amps = [-90, -70, -50, -30, -10]

    subthresh_hyperpol_dict, _ = fv.identify_subthreshold_hyperpol_with_amplitudes(features, sweeps)
    temp_data = fv.step_subthreshold(subthresh_hyperpol_dict, target_amps, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "step_subthresh.npy"))

    assert np.array_equal(test_data, temp_data)


def test_step_subthreshold_interpolation():
    target_amps = [-90, -70, -50, -30, -10]
    test_sweep_list = []
    test_amps = [-70, -30]
    t = np.arange(6)
    i = np.zeros_like(t)
    epochs = {"sweep": (0, 5), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    for a in test_amps:
        v = np.hstack([np.zeros(2), np.ones(2) * a, np.zeros(2)])
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        test_sweep_list.append(test_sweep)
    amp_sweep_dict = dict(zip(test_amps, test_sweep_list))
    output = fv.step_subthreshold(amp_sweep_dict, target_amps, start=2, end=4,
        extend_duration=1, subsample_interval=1)
    assert np.all(output[1:3] == -90)
    assert np.array_equal(output[4:8], test_sweep_list[0].v[1:-1])
    assert np.all(output[9:11] == -50)
    assert np.array_equal(output[12:16], test_sweep_list[1].v[1:-1])
    assert np.all(output[17:19] == -10)


def test_subthresh_norm_normalization():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    base = v[0]
    deflect_v = np.min(v)

    amp_sweep_dict = {-10: test_sweep}
    deflect_dict = {-10: (base, deflect_v)}
    output = fv.subthresh_norm(amp_sweep_dict, deflect_dict, start=t[0], end=t[-1],
        target_amp=-10, extend_duration=0, subsample_interval=1)
    assert np.isclose(output[0], 0)
    assert np.isclose(np.min(output), -1)


def test_subthresh_depol_norm_bad_steady_state_interval():
    with pytest.raises(ValueError):
        fv.subthresh_depol_norm({}, {}, start=0, end=1, steady_state_interval=2)


def test_subthresh_depol_norm_empty_result():
    output = fv.subthresh_depol_norm({}, {}, start=1.02, end=2.02)
    assert len(output) == 140
    assert np.all(np.isnan(output))


def test_subthresh_depol_norm_normalization():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    base = v[0]
    deflect_v = np.max(v)

    amp_sweep_dict = {10: test_sweep}
    deflect_dict = {10: (base, deflect_v)}

    output = fv.subthresh_depol_norm(amp_sweep_dict, deflect_dict, start=t[0], end=t[-1],
        steady_state_interval=1, subsample_interval=1, extend_duration=0)
    assert np.isclose(output[0], 0)
    assert np.isclose(output[-1], 1)

@pytest.mark.requires_inhouse_data
def test_subthresh_depol_norm(feature_vector_input):
    sweeps, features, start, end = feature_vector_input
    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_depol_with_amplitudes(features, sweeps)
    temp_data = fv.subthresh_depol_norm(amp_sweep_dict, deflect_dict, start, end)

    test_data = np.load(os.path.join(TEST_OUTPUT_DIR, "subthresh_depol_norm.npy"))

    assert np.array_equal(test_data, temp_data)


def test_first_ap_no_sweeps():
    ap_v, ap_dv = fv.first_ap_vectors([], [])
    assert np.all(ap_v == 0)


def test_first_ap_correct_section():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    test_spike_index = 10
    test_info = pd.DataFrame({"threshold_index": np.array([test_spike_index])})

    window_length = 5
    ap_v, ap_dv = fv.first_ap_vectors([test_sweep], [test_info],
        target_sampling_rate=sampling_rate, window_length=window_length)

    assert np.array_equal(ap_v, test_sweep.v[test_spike_index:test_spike_index + window_length])


def test_first_ap_resampling():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    test_spike_index = 10
    test_info = pd.DataFrame({"threshold_index": np.array([test_spike_index])})

    window_length = 10
    ap_v, ap_dv = fv.first_ap_vectors([test_sweep], [test_info],
        target_sampling_rate=sampling_rate / 2, window_length=window_length)

    assert ap_v.shape[0] == window_length / 2


def test_psth_sparse_firing():
    test_spike_times = [0.2, 0.5]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 50

    # All spikes are in own bins
    output = fv.psth_vector([spike_info], start=start, end=end, width=width)
    assert np.sum(output > 0) == len(test_spike_times)
    assert np.max(output) == 1 / (width * 0.001)


def test_psth_compressed_firing():
    test_spike_times = [0.2, 0.202, 0.204]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 50

    # All spikes are within one bin
    output = fv.psth_vector([spike_info], start=start, end=end, width=width)
    assert np.sum(output > 0) == 1
    assert np.max(output) == len(test_spike_times) / (width * 0.001)


def test_psth_number_of_spikes():
    np.random.seed(42)
    n_spikes = np.random.randint(0, 100)
    start = 1.02
    end = 2.02
    width = 50
    test_spike_times = np.random.random(n_spikes) * (end - start) + start
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})

    output = fv.psth_vector([spike_info], start=start, end=end, width=width)
    assert np.isclose(output.mean(), n_spikes)


def test_psth_between_sweep_interpolation():
    feature = "test_feature_name"
    test_spike_times = ([0.25, 0.75], [0.2, 0.5, 0.6, 0.7])
    available_list = []
    for tst in test_spike_times:
        spike_info = pd.DataFrame({
            "threshold_t": tst,
        })
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))

    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.psth_vector(si_list,
        start=start, end=end, width=width)

    assert np.array_equal(output[:n_bins], output[n_bins:2 * n_bins])
    assert np.array_equal(output[3 * n_bins:4 * n_bins], output[4 * n_bins:5 * n_bins])
    assert np.array_equal(output[2 * n_bins:3 * n_bins],
        np.vstack([output[n_bins:2 * n_bins], output[4 * n_bins:5 * n_bins]]).mean(axis=0))


def test_inst_freq_one_spike():
    test_spike_times = [0.2]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 20
    output = fv.inst_freq_vector([spike_info], start=start, end=end, width=width)
    assert np.all(output >= 1 / (end - start))


def test_inst_freq_initial_freq():
    test_spike_times = [0.2, 0.5, 0.6, 0.7]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 20
    output = fv.inst_freq_vector([spike_info], start=start, end=end, width=width)
    assert output[0] == 1. / (test_spike_times[0] - start)


def test_inst_freq_between_sweep_interpolation():
    feature = "test_feature_name"
    test_spike_times = ([0.25, 0.75], [0.2, 0.5, 0.6, 0.7])
    available_list = []
    for tst in test_spike_times:
        spike_info = pd.DataFrame({
            "threshold_t": tst,
        })
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))

    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.inst_freq_vector(si_list,
        start=start, end=end, width=width)

    assert np.array_equal(output[:n_bins], output[n_bins:2 * n_bins])
    assert np.array_equal(output[3 * n_bins:4 * n_bins], output[4 * n_bins:5 * n_bins])
    assert np.array_equal(output[2 * n_bins:3 * n_bins],
        np.vstack([output[n_bins:2 * n_bins], output[4 * n_bins:5 * n_bins]]).mean(axis=0))


def test_spike_feature_within_sweep_interpolation():
    feature = "test_feature_name"
    test_spike_times = [0.25, 0.75]
    test_feature_values = [1, 2]
    spike_info = pd.DataFrame({
        "threshold_t": test_spike_times,
        "clipped": np.zeros(2).astype(bool),
        feature: test_feature_values,
    })
    start = 0
    end = 1
    width = 20
    output = fv.spike_feature_vector(feature, [spike_info],
        start=start, end=end, width=width)
    assert output[0] == test_feature_values[0]
    assert output[len(output) // 2] > test_feature_values[0]
    assert output[len(output) // 2] < test_feature_values[-1]
    assert output[-1] == test_feature_values[-1]


def test_spike_feature_between_sweep_interpolation():
    feature = "test_feature_name"
    test_spike_times = [0.25, 0.75]
    test_feature_values = [1, 2]
    available_list = []
    for val in test_feature_values:
        spike_info = pd.DataFrame({
            "threshold_t": test_spike_times,
            "clipped": np.zeros(2).astype(bool),
            feature: [val] * 2,
        })
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))
    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.spike_feature_vector(feature, si_list,
        start=start, end=end, width=width)
    assert np.all(output[:n_bins] == test_feature_values[0])
    assert np.all(output[n_bins:2 * n_bins] == test_feature_values[0])
    assert np.all(output[2 * n_bins:3 * n_bins] == np.mean(test_feature_values))
    assert np.all(output[3 * n_bins:4 * n_bins] == test_feature_values[1])
    assert np.all(output[4 * n_bins:5 * n_bins] == test_feature_values[1])


def test_identify_sub_hyperpol_levels():
    test_input_amplitudes = [-2000, -100, -90, -50, -10, 10]
    test_features = {"subthreshold_sweeps": pd.DataFrame({
        "stim_amp": test_input_amplitudes,
        "peak_deflect": list(zip(np.zeros(len(test_input_amplitudes)),
            np.zeros(len(test_input_amplitudes)))),
        "v_baseline": np.ones(len(test_input_amplitudes)),
    })}

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_hyperpol_with_amplitudes(
        test_features, test_sweep_set)

    for k in amp_sweep_dict:
        assert k in deflect_dict
        assert len(deflect_dict[k]) == 2

    less_than_one_nanoamp = [a for a in test_input_amplitudes if a < -1000]
    for a in less_than_one_nanoamp:
        assert a not in amp_sweep_dict

    depolarizing = [a for a in test_input_amplitudes if a >= 0]
    for a in depolarizing:
        assert a not in amp_sweep_dict

    should_belong = [a for a in test_input_amplitudes if a >= 0 and a < -1000]
    for a in should_belong:
        assert a in amp_sweep_dict


def test_identify_sub_depol_levels_with_subthreshold_sweeps():
    test_input_amplitudes = [-50, -10, 10, 20]
    test_features = {"subthreshold_sweeps": pd.DataFrame({
        "stim_amp": test_input_amplitudes,
        "peak_deflect": list(zip(np.zeros(len(test_input_amplitudes)),
            np.zeros(len(test_input_amplitudes)))),
        "v_baseline": np.ones(len(test_input_amplitudes)),
    })}

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_depol_with_amplitudes(
        test_features, test_sweep_set)

    for k in amp_sweep_dict:
        assert k in deflect_dict
        assert len(deflect_dict[k]) == 2

    depolarizing = [a for a in test_input_amplitudes if a > 0]
    for a in depolarizing:
        assert a in amp_sweep_dict

    hyperpolarizing = [a for a in test_input_amplitudes if a <= 0]
    for a in hyperpolarizing:
        assert a not in amp_sweep_dict


def test_identify_sub_depol_levels_without_subthreshold_sweeps():
    test_input_amplitudes = [-50, -10, 10, 20]
    test_avg_rate = [0, 0, 0, 5]
    test_features = {"sweeps": pd.DataFrame({
        "stim_amp": test_input_amplitudes,
        "peak_deflect": list(zip(np.zeros(len(test_input_amplitudes)),
            np.zeros(len(test_input_amplitudes)))),
        "v_baseline": np.ones(len(test_input_amplitudes)),
        "avg_rate": test_avg_rate,
    })}

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_depol_with_amplitudes(
        test_features, test_sweep_set)

    for k in amp_sweep_dict:
        assert k in deflect_dict
        assert len(deflect_dict[k]) == 2

    depolarizing_spiking = [a for a, r in zip(test_input_amplitudes, test_avg_rate)
         if a > 0 and r > 0]
    for a in depolarizing_spiking:
        assert a not in amp_sweep_dict

    depolarizing_non_spiking = [a for a, r in zip(test_input_amplitudes, test_avg_rate)
         if a > 0 and r == 0]
    for a in depolarizing_non_spiking:
        assert a in amp_sweep_dict

    hyperpolarizing = [a for a in test_input_amplitudes if a <= 0]
    for a in hyperpolarizing:
        assert a not in amp_sweep_dict


def test_identify_isi_shape_min_spike():
    min_spike = 5

    test_input_amplitudes = [10, 20, 30, 40, 50, 60]
    test_avg_rate = [0, 1, 5, 5, 10, 20]
    test_features = {
        "sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {"sweep": (0, n_points - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike)

    assert np.array_equal(selected_sweep.v, sweep_list[2].v)


def test_identify_isi_shape_largest_below_min_spike():
    min_spike = 5

    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 3, 4]
    test_features = {
        "sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {"sweep": (0, n_points - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike)

    assert np.array_equal(selected_sweep.v, sweep_list[-1].v)


def test_identify_isi_shape_one_spike():
    min_spike = 5

    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 1, 1]
    test_features = {
        "sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {"sweep": (0, n_points - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike)

    assert np.array_equal(selected_sweep.v, sweep_list[1].v)


def test_identify_supra_no_tolerance():
    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "rheobase_i": 20,
    }

    target_amplitudes = np.array([0, 20])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=None, amp_tolerance=0)

    assert sweeps_to_use == [1, 3]


def test_identify_supra_tolerance():
    test_input_amplitudes = [10, 20, 30, 45]
    test_avg_rate = [0, 1, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "rheobase_i": 20,
    }

    target_amplitudes = np.array([0, 20])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=None, amp_tolerance=5)

    assert sweeps_to_use == [1, 3]


def test_identify_supra_shift():
    test_input_amplitudes = [10, 20, 40, 60]
    test_avg_rate = [1, 2, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame({
            "stim_amp": test_input_amplitudes,
            "avg_rate": test_avg_rate,
        }),
        "rheobase_i": 10,
    }

    target_amplitudes = np.array([0, 20, 40])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=10, amp_tolerance=5)

    assert sweeps_to_use == [1, 2, 3]


def test_isi_shape_aligned():
    # Random test sweep
    np.random.seed(42)
    v = np.random.randn(1000)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_threshold_index = np.array([100, 300, 500])
    test_fast_trough_index = test_threshold_index + 20
    test_threshold_v = np.random.randint(-100, -20, size=len(test_threshold_index))

    test_spike_info = pd.DataFrame({
        "threshold_index": test_threshold_index,
        "fast_trough_index": test_fast_trough_index,
        "threshold_v": test_threshold_v,
        "fast_trough_t": test_fast_trough_index,
    })

    n_points = 100
    isi_norm = fv.isi_shape(test_sweep, test_spike_info, end, n_points=n_points)
    assert len(isi_norm) == n_points
    assert isi_norm[0] == np.mean(test_sweep.v[test_fast_trough_index[:-1]] - test_threshold_v[:-1])


def test_isi_shape_aligned():
    # Random test sweep
    np.random.seed(42)
    v = np.random.randn(1000)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_threshold_index = np.array([100, 220, 340])
    test_fast_trough_index = test_threshold_index + 20
    test_threshold_v = np.random.randint(-100, -20, size=len(test_threshold_index))

    test_spike_info = pd.DataFrame({
        "threshold_index": test_threshold_index,
        "fast_trough_index": test_fast_trough_index,
        "threshold_v": test_threshold_v,
        "fast_trough_t": test_fast_trough_index,
    })

    n_points = 100
    isi_norm = fv.isi_shape(test_sweep, test_spike_info, end, n_points=n_points)
    assert len(isi_norm) == n_points
    assert isi_norm[0] == np.mean(test_sweep.v[test_fast_trough_index[:-1]] - test_threshold_v[:-1])


def test_isi_shape_skip_short():
    # Random test sweep
    np.random.seed(42)
    v = np.random.randn(1000)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_subsample = 3
    test_threshold_index = np.array([100, 130, 150 + 100 * test_subsample])
    test_fast_trough_index = test_threshold_index + 20
    test_threshold_v = np.random.randint(-100, -20, size=len(test_threshold_index))

    test_spike_info = pd.DataFrame({
        "threshold_index": test_threshold_index,
        "fast_trough_index": test_fast_trough_index,
        "threshold_v": test_threshold_v,
        "fast_trough_t": test_fast_trough_index,
    })

    n_points = 100
    isi_norm = fv.isi_shape(test_sweep, test_spike_info, end, n_points=n_points)
    assert len(isi_norm) == n_points

    # Should only use second ISI
    assert isi_norm[0] == (test_sweep.v[test_fast_trough_index[1]:test_fast_trough_index[1] + test_subsample].mean()
        - test_threshold_v[1])


def test_isi_shape_one_spike():
    # Test sweep
    np.random.seed(42)
    v = np.zeros(1000)
    v[100:400] = np.linspace(-30, 0, 300)
    print(v[280:300])
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {"sweep": (0, len(v) - 1), "test": None, "recording": None, "experiment": None, "stim": None}
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_threshold_index = [80]
    test_fast_trough_index = [100]
    test_threshold_v = [0]

    test_spike_info = pd.DataFrame({
        "threshold_index": test_threshold_index,
        "fast_trough_index": test_fast_trough_index,
        "threshold_v": test_threshold_v,
        "fast_trough_t": test_fast_trough_index,
    })

    n_points = 100
    isi_norm = fv.isi_shape(test_sweep, test_spike_info, end, n_points=n_points,
        steady_state_interval=10, single_max_duration=500)
    assert len(isi_norm) == n_points

    assert isi_norm[0] < 0
    assert isi_norm[0] >= -30
