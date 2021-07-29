import numpy as np
import pandas as pd
from dictdiffer import diff

import ipfx.feature_vectors as fv
from ipfx.stimulus import StimulusOntology
from ipfx.sweep import Sweep, SweepSet
import allensdk.core.json_utilities as ju
import pytest


ontology = StimulusOntology(
    ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))


@pytest.fixture
def subthreshold_sweeps():
    return {
        "index": [0, 1, 2, 3, 4],
        "columns": [
            "avg_rate",
            "peak_deflect",
            "stim_amp",
            "v_baseline",
            "sag",
            "adapt",
            "latency",
            "isi_cv",
            "mean_isi",
            "median_isi",
            "first_isi",
        ],
        "data": [
            [
                0.0,
                (-70.65, 35892),
                -29.999998092651367,
                -68.25825500488281,
                0.054702017456293106,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-72.2, 33470),
                -50.0,
                -68.39005279541016,
                0.01751580648124218,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-73.66875, 79087),
                -70.0,
                -68.42821502685547,
                0.004760173615068197,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-30.550001, 31322),
                469.9999694824219,
                -68.29730987548828,
                -8.956585884094238,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-28.54375, 31625),
                479.9999694824219,
                -68.26968383789062,
                -9.093791961669922,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        ],
    }


def test_identify_sweep_for_isi_shape():
    sweeps = {
        "index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "columns": [
            "avg_rate",
            "peak_deflect",
            "stim_amp",
            "v_baseline",
            "sag",
            "adapt",
            "latency",
            "isi_cv",
            "mean_isi",
            "median_isi",
            "first_isi",
        ],
        "data": [
            [
                0.0,
                (-70.65, 35892),
                -29.999998092651367,
                -68.25825500488281,
                0.054702017456293106,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-72.2, 33470),
                -50.0,
                -68.39005279541016,
                0.01751580648124218,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-73.66875, 79087),
                -70.0,
                -68.42821502685547,
                0.004760173615068197,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-30.550001, 31322),
                469.9999694824219,
                -68.29730987548828,
                -8.956585884094238,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                0.0,
                (-28.54375, 31625),
                479.9999694824219,
                -68.26968383789062,
                -9.093791961669922,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                5.000000000000001,
                (14.98125, 30920),
                489.9999694824219,
                -67.90441131591797,
                -8.981210708618164,
                0.00839006531426236,
                0.018079999999999985,
                0.026166652896782595,
                0.016109999999999985,
                0.01608000000000004,
                0.015880000000000005,
            ],
            [
                30.000000000000004,
                (15.19375, 32169),
                520.0,
                -67.4559097290039,
                -8.814754486083984,
                0.00801722722893242,
                0.01963999999999999,
                0.1548801847128805,
                0.015706896551724133,
                0.0156400000000001,
                0.011479999999999935,
            ],
            [
                72.00000000000001,
                (15.31875, 31168),
                540.0,
                -67.43457794189453,
                -6.910084247589111,
                0.001789278456852954,
                0.012699999999999934,
                0.09433347064888947,
                0.013758591549295774,
                0.013900000000000023,
                0.010319999999999996,
            ],
            [
                81.00000000000001,
                (15.7125, 52887),
                560.0,
                -67.80497741699219,
                -6.805542945861816,
                0.002374929272995164,
                0.011459999999999915,
                0.0921604402099859,
                0.0123335,
                0.012390000000000012,
                0.009160000000000057,
            ],
        ],
    }
    end = 1.5
    start = 0.5

    class SomeSweeps:
        @property
        def sweeps(self):
            ndata = len(sweeps["data"][0])
            nsweeps = len(sweeps["index"])
            return np.arange(nsweeps * ndata).reshape(nsweeps, ndata)

    isi_sweep, isi_sweep_spike_info = fv.identify_sweep_for_isi_shape(
        SomeSweeps(),
        {"sweeps": pd.DataFrame(**sweeps), "spikes_set": {5: "foo"}},
        end - start,
    )

    assert isi_sweep_spike_info == "foo"
    assert np.allclose(isi_sweep, np.arange(55, 66))


def test_isi_shape():

    sweep_spike_info = {
        "fast_trough_index": [0, 10, -10000],
        "threshold_index": [-10000, 10, 20],
        "threshold_v": [1, 2, -10000],
    }

    class Sweep:
        @property
        def v(self):
            return np.arange(20)

        @property
        def t(self):
            return np.arange(20)

    obtained = fv.isi_shape(
        Sweep(),
        pd.DataFrame(sweep_spike_info),
        50,
        n_points=10
    )
    assert np.allclose(np.arange(3.5, 13.5, 1.0), obtained)


def test_identify_subthreshold_hyperpol_with_amplitudes(subthreshold_sweeps):

    class SomeSweeps:
        @property
        def sweeps(self):
            ndata = len(subthreshold_sweeps["data"][0])
            nsweeps = len(subthreshold_sweeps["index"])
            return np.arange(nsweeps * ndata).reshape(nsweeps, ndata)

    obtained, _ = \
        fv.identify_subthreshold_hyperpol_with_amplitudes(
            {"subthreshold_sweeps": pd.DataFrame(**subthreshold_sweeps)},
            SomeSweeps()
        )

    expected = {
        -30.0: np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        -50.0: np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        -70.0: np.array([22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
    }

    differing = list(diff(expected, obtained))
    assert not differing, differing


def test_step_subthreshold():
    class Sweep:
        @property
        def v(self):
            return np.arange(10)

        @property
        def t(self):
            return np.arange(10)

    subthresh_hyperpol_dict = {
        -30.0: Sweep()
    }

    obtained = fv.step_subthreshold(
        subthresh_hyperpol_dict, [-30], 4, 6, subsample_interval=1)
    assert np.allclose(obtained, [4, 5])


def test_step_subthreshold_interpolation():
    target_amps = [-90, -70, -50, -30, -10]
    test_sweep_list = []
    test_amps = [-70, -30]
    t = np.arange(6)
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, 5),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    for a in test_amps:
        v = np.hstack([np.zeros(2), np.ones(2) * a, np.zeros(2)])
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        test_sweep_list.append(test_sweep)
    amp_sweep_dict = dict(zip(test_amps, test_sweep_list))
    output = fv.step_subthreshold(
        amp_sweep_dict,
        target_amps,
        start=2,
        end=4,
        extend_duration=1,
        subsample_interval=1,
    )
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
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    base = v[0]
    deflect_v = np.min(v)

    amp_sweep_dict = {-10: test_sweep}
    deflect_dict = {-10: (base, deflect_v)}
    output = fv.subthresh_norm(
        amp_sweep_dict,
        deflect_dict,
        start=t[0],
        end=t[-1],
        target_amp=-10,
        extend_duration=0,
        subsample_interval=1,
    )
    assert np.isclose(output[0], 0)
    assert np.isclose(np.min(output), -1)


def test_subthresh_depol_norm_bad_steady_state_interval():
    with pytest.raises(ValueError):
        fv.subthresh_depol_norm(
            {}, {}, start=0, end=1, steady_state_interval=2
        )


def test_subthresh_depol_norm_empty_result():
    output = fv.subthresh_depol_norm({}, {}, start=1.02, end=2.02)
    assert len(output) == 140
    assert np.all(np.isnan(output))


def test_subthresh_depol_norm_normalization():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    base = v[0]
    deflect_v = np.max(v)

    amp_sweep_dict = {10: test_sweep}
    deflect_dict = {10: (base, deflect_v)}

    output = fv.subthresh_depol_norm(
        amp_sweep_dict,
        deflect_dict,
        start=t[0],
        end=t[-1],
        steady_state_interval=1,
        subsample_interval=1,
        extend_duration=0,
    )
    assert np.isclose(output[0], 0)
    assert np.isclose(output[-1], 1)



def test_identify_subthreshold_depol_with_amplitudes(subthreshold_sweeps):

    class SomeSweeps:
        @property
        def sweeps(self):
            ndata = len(subthreshold_sweeps["data"][0])
            nsweeps = len(subthreshold_sweeps["index"])
            return np.arange(nsweeps * ndata).reshape(nsweeps, ndata)

    amp, deflect = fv.identify_subthreshold_depol_with_amplitudes(
        {"subthreshold_sweeps": pd.DataFrame(**subthreshold_sweeps)},
        SomeSweeps()
    )

    expected = {
        "amp": {
            470.0: np.array([33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]),
            480.0: np.array([44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54])
        },
        "deflect": {
            470.0: (-68.29730987548828, -30.550001),
            480.0: (-68.26968383789062, -28.54375)
        }
    }
    obtained = {"amp": amp, "deflect": deflect}
    diffs = list(diff(obtained, expected))
    assert not diffs, diffs


def test_subthresh_depol_norm():

    class Sweep:
        @property
        def v(self):
            return np.arange(10)

        @property
        def t(self):
            return np.arange(10)

    amp_sweep_dict = {50: Sweep()}
    deflect_dict = {50: (1, 2)}

    obtained = fv.subthresh_depol_norm(
        amp_sweep_dict, deflect_dict, 4, 7, subsample_interval=1,
        steady_state_interval=2
    )
    assert np.allclose([2/3, 8/9, 10/9], obtained)


def test_first_ap_no_sweeps():
    ap_v, ap_dv = fv.first_ap_vectors([], [])
    assert np.all(ap_v == 0)


def test_first_ap_correct_section():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    test_spike_index = 10
    test_info = pd.DataFrame({"threshold_index": np.array([test_spike_index])})

    window_length = 5
    ap_v, ap_dv = fv.first_ap_vectors(
        [test_sweep],
        [test_info],
        target_sampling_rate=sampling_rate,
        window_length=window_length,
    )

    assert np.array_equal(
        ap_v, test_sweep.v[test_spike_index : test_spike_index + window_length]
    )


def test_first_ap_resampling():
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)

    test_spike_index = 10
    test_info = pd.DataFrame({"threshold_index": np.array([test_spike_index])})

    window_length = 10
    ap_v, ap_dv = fv.first_ap_vectors(
        [test_sweep],
        [test_info],
        target_sampling_rate=sampling_rate / 2,
        window_length=window_length,
    )

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
    test_spike_times = ([0.25, 0.75], [0.2, 0.5, 0.6, 0.7])
    available_list = []
    for tst in test_spike_times:
        spike_info = pd.DataFrame({"threshold_t": tst,})
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))

    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.psth_vector(si_list, start=start, end=end, width=width)

    assert np.array_equal(output[:n_bins], output[n_bins : 2 * n_bins])
    assert np.array_equal(
        output[3 * n_bins : 4 * n_bins], output[4 * n_bins : 5 * n_bins]
    )
    assert np.array_equal(
        output[2 * n_bins : 3 * n_bins],
        np.vstack([output[n_bins : 2 * n_bins], output[4 * n_bins : 5 * n_bins]]).mean(
            axis=0
        ),
    )


def test_psth_duration_rounding():
    start_a, end_a = 1.02, 2.02
    start_b, end_b = 1.02, 2.0199999999999996

    np.random.seed(42)
    n_spikes = np.random.randint(0, 100)

    test_spike_times = np.random.random(n_spikes) * (end_a - start_a) + start_a
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})

    width = 50

    output_a = fv.psth_vector([spike_info], start=start_a, end=end_a, width=width)
    output_b = fv.psth_vector([spike_info], start=start_b, end=end_b, width=width)

    assert output_a.shape == output_b.shape


def test_inst_freq_one_spike():
    test_spike_times = [0.2]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 20
    output = fv.inst_freq_vector(
        [spike_info], start=start, end=end, width=width
    )
    assert np.all(output >= 1 / (end - start))


def test_inst_freq_initial_freq():
    test_spike_times = [0.2, 0.5, 0.6, 0.7]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})
    start = 0
    end = 1
    width = 20
    output = fv.inst_freq_vector([spike_info], start=start, end=end, width=width)
    assert output[0] == 1.0 / (test_spike_times[0] - start)


def test_inst_freq_between_sweep_interpolation():
    test_spike_times = ([0.25, 0.75], [0.2, 0.5, 0.6, 0.7])
    available_list = []
    for tst in test_spike_times:
        spike_info = pd.DataFrame({"threshold_t": tst,})
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))

    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.inst_freq_vector(si_list, start=start, end=end, width=width)

    assert np.array_equal(output[:n_bins], output[n_bins : 2 * n_bins])
    assert np.array_equal(
        output[3 * n_bins : 4 * n_bins], output[4 * n_bins : 5 * n_bins]
    )
    assert np.array_equal(
        output[2 * n_bins : 3 * n_bins],
        np.vstack([output[n_bins : 2 * n_bins], output[4 * n_bins : 5 * n_bins]]).mean(
            axis=0
        ),
    )


def test_inst_freq_duration_rounding():
    start_a, end_a = 1.02, 2.02
    start_b, end_b = 1.02, 2.0199999999999996

    test_spike_times = [1.2, 1.5, 1.6, 1.7]
    spike_info = pd.DataFrame({"threshold_t": test_spike_times})

    width = 20

    output_a = fv.inst_freq_vector([spike_info], start=start_a, end=end_a, width=width)
    output_b = fv.inst_freq_vector([spike_info], start=start_b, end=end_b, width=width)

    assert output_a.shape == output_b.shape


def test_spike_feature_within_sweep_interpolation():
    feature = "test_feature_name"
    test_spike_times = [0.25, 0.75]
    test_feature_values = [1, 2]
    spike_info = pd.DataFrame(
        {
            "threshold_t": test_spike_times,
            "clipped": np.zeros(2).astype(bool),
            feature: test_feature_values,
        }
    )
    start = 0
    end = 1
    width = 20
    output = fv.spike_feature_vector(
        feature, [spike_info], start=start, end=end, width=width
    )
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
        spike_info = pd.DataFrame(
            {
                "threshold_t": test_spike_times,
                "clipped": np.zeros(2).astype(bool),
                feature: [val] * 2,
            }
        )
        available_list.append(spike_info)

    start = 0
    end = 1
    width = 20
    n_bins = int((end - start) / (width * 0.001))
    si_list = [None, available_list[0], None, available_list[1], None]
    output = fv.spike_feature_vector(
        feature, si_list, start=start, end=end, width=width
    )
    assert np.all(output[:n_bins] == test_feature_values[0])
    assert np.all(output[n_bins : 2 * n_bins] == test_feature_values[0])
    assert np.all(output[2 * n_bins : 3 * n_bins] == np.mean(test_feature_values))
    assert np.all(output[3 * n_bins : 4 * n_bins] == test_feature_values[1])
    assert np.all(output[4 * n_bins : 5 * n_bins] == test_feature_values[1])


def test_spike_feature_duration_rounding():
    start_a, end_a = 1.02, 2.02
    start_b, end_b = 1.02, 2.0199999999999996

    test_spike_times = [1.2, 1.5, 1.6, 1.7]

    feature = "test_feature_name"
    test_feature_values = [1, 2, 3, 4]

    spike_info = pd.DataFrame({
        "threshold_t": test_spike_times,
        "clipped": np.zeros_like(test_spike_times).astype(bool),
        feature: test_feature_values,
    })

    width = 20

    output_a = fv.spike_feature_vector(
        feature, [spike_info], start=start_a, end=end_a, width=width)
    output_b = fv.spike_feature_vector(
        feature, [spike_info], start=start_b, end=end_b, width=width)

    assert output_a.shape == output_b.shape


def test_identify_sub_hyperpol_levels():
    test_input_amplitudes = [-2000, -100, -90, -50, -10, 10]
    test_features = {
        "subthreshold_sweeps": pd.DataFrame(
            {
                "stim_amp": test_input_amplitudes,
                "peak_deflect": list(
                    zip(
                        np.zeros(len(test_input_amplitudes)),
                        np.zeros(len(test_input_amplitudes)),
                    )
                ),
                "v_baseline": np.ones(len(test_input_amplitudes)),
            }
        )
    }

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_hyperpol_with_amplitudes(
        test_features, test_sweep_set
    )

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
    test_features = {
        "subthreshold_sweeps": pd.DataFrame(
            {
                "stim_amp": test_input_amplitudes,
                "peak_deflect": list(
                    zip(
                        np.zeros(len(test_input_amplitudes)),
                        np.zeros(len(test_input_amplitudes)),
                    )
                ),
                "v_baseline": np.ones(len(test_input_amplitudes)),
            }
        )
    }

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_depol_with_amplitudes(
        test_features, test_sweep_set
    )

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
    test_features = {
        "sweeps": pd.DataFrame(
            {
                "stim_amp": test_input_amplitudes,
                "peak_deflect": list(
                    zip(
                        np.zeros(len(test_input_amplitudes)),
                        np.zeros(len(test_input_amplitudes)),
                    )
                ),
                "v_baseline": np.ones(len(test_input_amplitudes)),
                "avg_rate": test_avg_rate,
            }
        )
    }

    # Random test sweeps
    np.random.seed(42)
    v = np.random.randn(100)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    amp_sweep_dict, deflect_dict = fv.identify_subthreshold_depol_with_amplitudes(
        test_features, test_sweep_set
    )

    for k in amp_sweep_dict:
        assert k in deflect_dict
        assert len(deflect_dict[k]) == 2

    depolarizing_spiking = [
        a for a, r in zip(test_input_amplitudes, test_avg_rate) if a > 0 and r > 0
    ]
    for a in depolarizing_spiking:
        assert a not in amp_sweep_dict

    depolarizing_non_spiking = [
        a for a, r in zip(test_input_amplitudes, test_avg_rate) if a > 0 and r == 0
    ]
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
        "sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, n_points - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike
    )

    assert np.array_equal(selected_sweep.v, sweep_list[2].v)


def test_identify_isi_shape_largest_below_min_spike():
    min_spike = 5

    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 3, 4]
    test_features = {
        "sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, n_points - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike
    )

    assert np.array_equal(selected_sweep.v, sweep_list[-1].v)


def test_identify_isi_shape_one_spike():
    min_spike = 5

    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 1, 1]
    test_features = {
        "sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "spikes_set": [None] * len(test_avg_rate),
    }

    # Random test sweeps
    np.random.seed(42)
    n_points = 100
    t = np.arange(n_points)
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, n_points - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    sweep_list = []
    for a in test_input_amplitudes:
        v = np.random.randn(n_points)
        test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
        sweep_list.append(test_sweep)
    test_sweep_set = SweepSet(sweep_list)

    selected_sweep, _ = fv.identify_sweep_for_isi_shape(
        test_sweep_set, test_features, duration=1, min_spike=min_spike
    )

    assert np.array_equal(selected_sweep.v, sweep_list[1].v)


def test_identify_supra_no_tolerance():
    test_input_amplitudes = [10, 20, 30, 40]
    test_avg_rate = [0, 1, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "rheobase_i": 20,
    }

    target_amplitudes = np.array([0, 20])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=None, amp_tolerance=0
    )

    assert sweeps_to_use == [1, 3]


def test_identify_supra_tolerance():
    test_input_amplitudes = [10, 20, 30, 45]
    test_avg_rate = [0, 1, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "rheobase_i": 20,
    }

    target_amplitudes = np.array([0, 20])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=None, amp_tolerance=5
    )

    assert sweeps_to_use == [1, 3]


def test_identify_supra_shift():
    test_input_amplitudes = [10, 20, 40, 60]
    test_avg_rate = [1, 2, 5, 10]
    test_features = {
        "spiking_sweeps": pd.DataFrame(
            {"stim_amp": test_input_amplitudes, "avg_rate": test_avg_rate,}
        ),
        "rheobase_i": 10,
    }

    target_amplitudes = np.array([0, 20, 40])
    sweeps_to_use = fv._identify_suprathreshold_indices(
        test_features, target_amplitudes, shift=10, amp_tolerance=5
    )

    assert sweeps_to_use == [1, 2, 3]


def test_isi_shape_aligned():
    # Random test sweep
    np.random.seed(42)
    v = np.random.randn(1000)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_threshold_index = np.array([100, 220, 340])
    test_fast_trough_index = test_threshold_index + 20
    test_threshold_v = np.random.randint(
        -100, -20, size=len(test_threshold_index)
    )

    test_spike_info = pd.DataFrame(
        {
            "threshold_index": test_threshold_index,
            "fast_trough_index": test_fast_trough_index,
            "threshold_v": test_threshold_v,
            "fast_trough_t": test_fast_trough_index,
        }
    )

    n_points = 100
    isi_norm = fv.isi_shape(
        test_sweep, test_spike_info, end, n_points=n_points
    )
    assert len(isi_norm) == n_points
    assert isi_norm[0] == np.mean(
        test_sweep.v[test_fast_trough_index[:-1]] - test_threshold_v[:-1]
    )


def test_isi_shape_skip_short():
    # Random test sweep
    np.random.seed(42)
    v = np.random.randn(1000)
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
    sampling_rate = 1
    clamp_mode = "CurrentClamp"
    test_sweep = Sweep(t, v, i, clamp_mode, sampling_rate, epochs=epochs)
    end = t[-100]

    test_subsample = 3
    test_threshold_index = np.array([100, 130, 150 + 100 * test_subsample])
    test_fast_trough_index = test_threshold_index + 20
    test_threshold_v = np.random.randint(
        -100, -20, size=len(test_threshold_index)
    )

    test_spike_info = pd.DataFrame(
        {
            "threshold_index": test_threshold_index,
            "fast_trough_index": test_fast_trough_index,
            "threshold_v": test_threshold_v,
            "fast_trough_t": test_fast_trough_index,
        }
    )

    n_points = 100
    isi_norm = fv.isi_shape(
        test_sweep, test_spike_info, end, n_points=n_points
    )
    assert len(isi_norm) == n_points

    # Should only use second ISI
    assert isi_norm[0] == (
        test_sweep.v[
            test_fast_trough_index[1] : test_fast_trough_index[1] + test_subsample
        ].mean()
        - test_threshold_v[1]
    )


def test_isi_shape_one_spike():
    # Test sweep
    np.random.seed(42)
    v = np.zeros(1000)
    v[100:400] = np.linspace(-30, 0, 300)
    print(v[280:300])
    t = np.arange(len(v))
    i = np.zeros_like(t)
    epochs = {
        "recording": (0, len(v) - 1),
        "test": None,
        "experiment": None,
        "stim": None,
    }
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
    isi_norm = fv.isi_shape(
        test_sweep, test_spike_info, end, n_points=n_points,
        steady_state_interval=10, single_max_duration=500
    )
    assert len(isi_norm) == n_points

    assert isi_norm[0] < 0
    assert isi_norm[0] >= -30
