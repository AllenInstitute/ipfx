from builtins import zip
import ipfx.qc_features as qcf
import numpy as np
import pytest

def test_measure_blowout():
    a = np.array([0, 0, 1, 1])
    b = qcf.measure_blowout(a, 0)
    assert b == 0.5

    b = qcf.measure_blowout(a, 2)
    assert b == 1.0


def test_measure_electrode_0():
    a = np.array([1, 1, 1, 1])
    b = qcf.measure_electrode_0(a, 1)
    assert b is None

    b = qcf.measure_electrode_0(a, 1000)
    assert b == 1


def test_measure_seal():
    # measure_seal now averages a series of square voltage test pulses (skipping
    # the first) and fits the capacitive transient to recover the steady-state
    # resistance. Build a cell-attached-style recording: v is a clean square
    # command (mV), curr is a step with a decaying capacitive transient (pA).
    dt = 1e-5
    pulse_dur_pts = 500   # 5 ms
    gap_pts = 400         # 4 ms between pulses
    lead_pts = 400
    n_pulses = 4          # first is treated as the test pulse and skipped

    delta_v_mV = 5.0
    r_seal = 1e9  # 1 GOhm
    i_ss_pA = (delta_v_mV * 1e-3 / r_seal) * 1e12  # steady current, pA
    peak_extra_pA = 200.0
    tau = 5e-5

    total = lead_pts + n_pulses * (pulse_dur_pts + gap_pts)
    v = np.zeros(total)
    curr = np.zeros(total)
    t_rel = np.arange(pulse_dur_pts) * dt
    idx = lead_pts
    for _ in range(n_pulses):
        up = idx
        down = idx + pulse_dur_pts
        v[up:down] = delta_v_mV
        curr[up:down] = i_ss_pA + peak_extra_pA * np.exp(-t_rel / tau)
        idx = down + gap_pts
    t = dt * np.arange(total)

    b = qcf.measure_seal(v, curr, t)
    assert np.allclose([b], [1.0], rtol=1e-3)


def test_measure_input_resistance():
    # get_r_from_stable_pulse_response_fit now operates on a single averaged
    # pulse (avg_v in V, avg_i in A) plus the relative up/down indices, and fits
    # the capacitive transient to estimate the steady-state resistance.
    dt = 1e-5
    n = 1000
    up_ind = 200
    down_ind = 800

    delta_v = 5e-3        # V
    r_expected = 50e6     # Ohm (50 MOhm)
    i_ss = delta_v / r_expected  # steady-state current, A
    peak_extra = 200e-12
    tau = 5e-5

    avg_v = np.zeros(n)
    avg_v[up_ind:down_ind] = delta_v

    avg_i = np.zeros(n)
    t_rel = np.arange(down_ind - up_ind) * dt
    avg_i[up_ind:down_ind] = i_ss + peak_extra * np.exp(-t_rel / tau)

    t = dt * np.arange(n)

    r = qcf.get_r_from_stable_pulse_response_fit(
        avg_v, avg_i, t, up_ind, down_ind, post_transient_shift_ms=1.0)
    assert np.isclose(r, r_expected, rtol=1e-3)


def test_get_square_pulse_idx():

        v = [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0]
        up_idx = [6,11,15]
        down_idx = [7,12,17]

        assert up_idx,down_idx == qcf.get_square_pulse_idx(v)


def test_truncated_pulse():

    v = [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 2, 2, 2]

    with pytest.raises(AssertionError, match="Truncated square pulse"):
        qcf.get_square_pulse_idx(v)


def test_negative_pulse():

    v = [0, 0, 1, 1, 0, 0, -2, -2, 0, 0, 0, 0, 2, 2, 2, 0, 0]

    with pytest.raises(AssertionError, match="Negative square pulse"):
        qcf.get_square_pulse_idx(v)
