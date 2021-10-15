from ipfx.sweep import Sweep
import pytest
import numpy as np
import ipfx.qc_features as qcf
import ipfx.qc_feature_extractor as qfex
import ipfx.qc_feature_evaluator as qfev


vm_delta_mv_max = 1.0
i = [0,0,10,10,0,0,0,100,100,100,100,100,0,0,0,0]
passing_v = [-59.02,-59.02,-55,-55,-59.02,-59.02,-59.02,-20,20,-20,20,-20,-60.99,-60.99,-60.99,-60.99]
failing_v = [-58.98,-58.98,-55,-55,-58.98,-58.98,-58.98,-20,20,-20,20,-20,-61.01,-61.01,-61.01,-61.01]

sampling_rate = 2
autobias_v = -60
dt = 1./sampling_rate
t = np.arange(0,len(i))*dt  

passing_sweep = Sweep(t, passing_v, i, "CurrentClamp", sampling_rate, sweep_number=1, autobias_v=autobias_v)
failing_sweep = Sweep(t, failing_v, i, "CurrentClamp", sampling_rate, sweep_number=1, autobias_v=autobias_v)  
no_autobias_passing_sweep = Sweep(t, passing_v, i, "CurrentClamp", sampling_rate, sweep_number=1, autobias_v=None)
no_autobias_failing_sweep = Sweep(t, failing_v, i, "CurrentClamp", sampling_rate, sweep_number=1, autobias_v=None)

passing_mean_first_stability_epoch, _ = qcf.measure_vm(passing_sweep.v[4:7])
passing_mean_last_stability_epoch, _ = qcf.measure_vm(passing_sweep.v[12:16])

failing_mean_first_stability_epoch, _ = qcf.measure_vm(failing_sweep.v[4:7])
failing_mean_last_stability_epoch, _ = qcf.measure_vm(failing_sweep.v[12:16])

no_autobias_passing_mean_first_stability_epoch, _ = qcf.measure_vm(no_autobias_passing_sweep.v[4:7])
no_autobias_passing_mean_last_stability_epoch, _ = qcf.measure_vm(no_autobias_passing_sweep.v[12:16])

no_autobias_failing_mean_first_stability_epoch, _ = qcf.measure_vm(no_autobias_failing_sweep.v[4:7])
no_autobias_failing_mean_last_stability_epoch, _ = qcf.measure_vm(no_autobias_failing_sweep.v[12:16])

passing_qc_features = qfex.current_clamp_sweep_qc_features(passing_sweep, False)
failing_qc_features = qfex.current_clamp_sweep_qc_features(failing_sweep, False)
no_autobias_passing_qc_features = qfex.current_clamp_sweep_qc_features(no_autobias_passing_sweep, False)
no_autobias_failing_qc_features = qfex.current_clamp_sweep_qc_features(no_autobias_failing_sweep, False)
ramp_failing_qc_features = qfex.current_clamp_sweep_qc_features(failing_sweep, True)
ramp_no_autobias_failing_qc_features = qfex.current_clamp_sweep_qc_features(no_autobias_failing_sweep, True)

def test_sweep_autobias_v():

    assert passing_sweep.autobias_v == autobias_v
    assert failing_sweep.autobias_v == autobias_v
    assert no_autobias_passing_sweep.autobias_v == None


def test_measure_vm_delta():

    assert round(qcf.measure_vm_delta(passing_mean_first_stability_epoch, passing_sweep.autobias_v), 2) == 0.98
    assert round(qcf.measure_vm_delta(passing_mean_last_stability_epoch, passing_sweep.autobias_v), 2) == 0.99

    assert round(qcf.measure_vm_delta(failing_mean_first_stability_epoch, failing_sweep.autobias_v), 2) == 1.02
    assert round(qcf.measure_vm_delta(failing_mean_last_stability_epoch, failing_sweep.autobias_v), 2) == 1.01

    assert round(qcf.measure_vm_delta(no_autobias_passing_mean_first_stability_epoch, no_autobias_passing_mean_last_stability_epoch), 2) == 1.97  
    assert round(qcf.measure_vm_delta(no_autobias_failing_mean_first_stability_epoch, no_autobias_failing_mean_last_stability_epoch), 2) == 2.03 


def test_current_clamp_sweep_qc_features():
    
    assert passing_qc_features["vm_autobias_delta_mv"] < vm_delta_mv_max
    assert passing_qc_features["vm_delta_mv"] < (2 * vm_delta_mv_max)

    assert failing_qc_features["vm_autobias_delta_mv"] > vm_delta_mv_max
    assert failing_qc_features["vm_delta_mv"] > (2 * vm_delta_mv_max)

    assert no_autobias_passing_qc_features["vm_delta_mv"] < (2 * vm_delta_mv_max)
    assert no_autobias_passing_qc_features["vm_autobias_delta_mv"] == None

    assert no_autobias_failing_qc_features["vm_delta_mv"] > (2 * vm_delta_mv_max)
    assert no_autobias_failing_qc_features["vm_autobias_delta_mv"] == None

    assert ramp_failing_qc_features["vm_delta_mv"] == None
    assert ramp_failing_qc_features["vm_autobias_delta_mv"] == None

    assert ramp_no_autobias_failing_qc_features["vm_delta_mv"] == None
    assert ramp_no_autobias_failing_qc_features["vm_autobias_delta_mv"] == None


def test_qc_current_clamp_sweep():

    sweep_num = 1
    stim_name = "test_stimulus"

    passing_qc_features["sweep_number"] = sweep_num
    failing_qc_features["sweep_number"] = sweep_num
    no_autobias_passing_qc_features["sweep_number"] = sweep_num
    no_autobias_failing_qc_features["sweep_number"] = sweep_num
    ramp_failing_qc_features["sweep_number"] = sweep_num
    ramp_no_autobias_failing_qc_features["sweep_number"] = sweep_num
    
    passing_qc_features["stimulus_name"] = stim_name
    failing_qc_features["stimulus_name"] = stim_name
    no_autobias_passing_qc_features["stimulus_name"] = stim_name
    no_autobias_failing_qc_features["stimulus_name"] = stim_name
    ramp_failing_qc_features["stimulus_name"] = stim_name
    ramp_no_autobias_failing_qc_features["stimulus_name"] = stim_name

    passing_fail_tags = qfev.qc_current_clamp_sweep(passing_qc_features, False)
    failing_fail_tags = qfev.qc_current_clamp_sweep(failing_qc_features, False)
    no_autobias_passing_fail_tags = qfev.qc_current_clamp_sweep(no_autobias_passing_qc_features, False)
    no_autobias_failing_fail_tags = qfev.qc_current_clamp_sweep(no_autobias_failing_qc_features, False)
    ramp_failing_fail_tags = qfev.qc_current_clamp_sweep(ramp_failing_qc_features, True)
    ramp_no_autobias_failing_fail_tags = qfev.qc_current_clamp_sweep(ramp_no_autobias_failing_qc_features, True)

    assert passing_fail_tags == []
    assert failing_fail_tags == ["Vm delta: 1.020 above threshold: 1.000"]

    assert no_autobias_passing_fail_tags == []
    assert no_autobias_failing_fail_tags == ["Vm delta: 2.030 above threshold: 2.000"]

    assert ramp_failing_fail_tags == []
    assert ramp_no_autobias_failing_fail_tags == []

