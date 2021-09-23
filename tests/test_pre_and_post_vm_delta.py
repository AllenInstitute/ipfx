from ipfx.sweep import Sweep
import pytest
import numpy as np
import ipfx.qc_features as qcf
import ipfx.qc_feature_extractor as qfex
import ipfx.qc_feature_evaluator as qfev


vm_delta_mv_max = 1.0
i = [0,0,10,10,0,0,0,100,100,100,100,100,0,0,0,0]
passing_v = [-59.01,-59.01,-55,-55,-59.01,-59.01,-59.01,-20,20,-20,20,-20,-60.99,-60.99,-60.99,-60.99]
failing_v = [-58.99,-58.99,-55,-55,-58.99,-58.99,-58.99,-20,20,-20,20,-20,-61.01,-61.01,-61.01,-61.01]

sampling_rate = 2
autobias_v = -60
dt = 1./sampling_rate
t = np.arange(0,len(i))*dt  

passing_sweep = Sweep(t, passing_v, i, sampling_rate=sampling_rate, autobias_v=autobias_v, clamp_mode="CurrentClamp")
failing_sweep = Sweep(t, failing_v, i, sampling_rate=sampling_rate, autobias_v=autobias_v, clamp_mode="CurrentClamp")  

passing_mean_first_stability_epoch, _ = qcf.measure_vm(passing_v[4:7])
passing_mean_last_stability_epoch, _ = qcf.measure_vm(passing_v[12:16])

failing_mean_first_stability_epoch, _ = qcf.measure_vm(failing_v[4:7])
failing_mean_last_stability_epoch, _ = qcf.measure_vm(failing_v[12:16])

passing_qc_features = qfex.current_clamp_sweep_qc_features(passing_sweep, False)
failing_qc_features = qfex.current_clamp_sweep_qc_features(failing_sweep, False)


def test_sweep_autobias_v():

    assert passing_sweep.autobias_v == autobias_v
    assert failing_sweep.autobias_v == autobias_v


def test_measure_vm_delta():

    assert round(qcf.measure_vm_delta(passing_mean_first_stability_epoch, passing_sweep.autobias_v), 2) == 0.99
    assert round(qcf.measure_vm_delta(passing_mean_last_stability_epoch, passing_sweep.autobias_v), 2) == 0.99

    assert round(qcf.measure_vm_delta(failing_mean_first_stability_epoch, failing_sweep.autobias_v), 2) == 1.01
    assert round(qcf.measure_vm_delta(failing_mean_last_stability_epoch, failing_sweep.autobias_v), 2) == 1.01


def test_current_clamp_qc_features():
    
    assert passing_qc_features["pre_vm_delta_mv"] < vm_delta_mv_max
    assert passing_qc_features["post_vm_delta_mv"] < vm_delta_mv_max

    assert failing_qc_features["pre_vm_delta_mv"] > vm_delta_mv_max
    assert failing_qc_features["post_vm_delta_mv"] > vm_delta_mv_max


def test_qc_current_clamp_sweep():

    failing_qc_features["sweep_number"] = 1
    failing_qc_features["stimulus_name"] = "test_stimulus"

    fail_tags = qfev.qc_current_clamp_sweep(failing_qc_features, False)
    
    assert fail_tags[0] == "pre Vm delta: 1.010 above threshold:1.000"
    assert fail_tags[1] == "post Vm delta: 1.010 above threshold:1.000"


def test_none_autobias_v():

    none_autobias_v = None
    sweep = Sweep(t, passing_v, i, sampling_rate=sampling_rate, autobias_v=none_autobias_v, clamp_mode="CurrentClamp")
    delta = qcf.measure_vm_delta(passing_mean_first_stability_epoch, sweep.autobias_v)
    
    assert delta is None

