import os, json
import logging
import numpy as np
import qc_features as qcf


DEFAULT_QC_CRITERIA_FILE = os.path.join(os.path.dirname(__file__), 'qc_criteria.json')

def load_default_qc_criteria():
    logging.debug("loading default qc criteria file: %s", DEFAULT_QC_CRITERIA_FILE)
    with open(DEFAULT_QC_CRITERIA_FILE,"r") as f:
        return json.load(f)


def qc_experiment(data_set, cell_features, sweep_features, qc_criteria=None):

    """

    Parameters
    ----------
    data_set : AibsDataSet object
        dataset
    cell_features : dict
        cell features
    sweep_features: list of dicts
        sweep features
    qc_criteria : dict
        qc criteria

    Returns
    -------
        cell_state : list
        sweep_states : list
    """
    if qc_criteria is None:
        qc_criteria = load_default_qc_criteria()

    cell_state = qc_cell(cell_features, qc_criteria)

    sweep_data_index = { sweep['sweep_number']:sweep for sweep in sweep_features }

    sweep_states = []
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True,
                                                  exclude_test=True,
                                                  exclude_search=True)

    for sweep_num in iclamp_sweeps.sweep_number:
        sweep = sweep_data_index[sweep_num]
        failed, fail_tags = qc_current_clamp_sweep(data_set, sweep, qc_criteria)
        sweep_state = { 'sweep_number': sweep_num, 'passed': not failed, 'reasons': fail_tags }
        if failed:
            print (sweep_num, sweep["stimulus_name"], fail_tags)
        sweep_states.append(sweep_state)

    return cell_state, sweep_states


def qc_cell(cell_data, qc_criteria=None):
    """Evaluate cell state across different types of stimuli

    Parameters
    ----------
    cell_data : dict
        cell features
    qc_criteria : dict
        qc criteria

    Returns
    -------
        dict
            cell state
    """

    if qc_criteria is None:
        qc_criteria = load_default_qc_criteria()

    # PBS-333
    # C1NSSEED stimuli have many instances, but these instances aren't
    #   stored with the sweep. Ie, the sweep stores the stimulus value
    #   C1NSSEED while the stimulus table stores C1NSSEED_2150112.
    # To address this, check the stimulus table for any instance of
    #   C1NSSEED, and if it exists, append a plain-jane "C1NSSEED" stimulus
    #   so later checks work
    #
    #for name in jin["current_clamp_stimuli"]:
    #    if name.startswith("C1NSSEED_"):
    #        jin["current_clamp_stimuli"].append("C1NSSEED")
    cell_fail_tags = []

    cell_state = {}

    # blowout voltage
    cell_state["failed_blowout"] = evaluate_blowout(cell_data.get('blowout_mv', None),
                                                    qc_criteria['blowout_mv_min'],
                                                    qc_criteria['blowout_mv_max'],
                                                    cell_fail_tags)

    # "electrode 0"
    cell_state["failed_electrode_0"] = evaluate_electrode_0(cell_data.get('electrode_0_pa', None),
                                                            qc_criteria['electrode_0_pa_max'],
                                                            cell_fail_tags)

    # measure clamp seal
    cell_state["failed_seal"] = evaluate_seal(cell_data.get('seal_gohm', None),
                                              qc_criteria['seal_gohm_min'],
                                              cell_fail_tags)

    # input and access resistance
    cell_state["failed_input_access_resistance"] = \
        evaluate_input_and_access_resistance(cell_data.get("input_access_resistance_ratio", None),
                                             qc_criteria["input_vs_access_resistance_max"],
                                             cell_data.get("initial_access_resistance_mohm", None),
                                             qc_criteria["access_resistance_mohm_min"],
                                             qc_criteria["access_resistance_mohm_max"],
                                             cell_fail_tags)

    cell_state["fail_tags"] = cell_fail_tags

    return cell_state


def qc_current_clamp_sweep(data_set, sweep, qc_criteria=None):
    """QC for the current-clamp sweeps

    Parameters
    ----------
    data_set : MiesDataSet
        data set
    sweep : dict
        features of a sweep
    qc_criteria : dict
        qc criteria

    Returns
    -------
        fails   : int
            number of fails
        fail_tags : list of str
            tags of the failed sweeps

    """
    if qc_criteria is None:
        qc_criteria = qcf.load_default_qc_criteria()

    # keep track of failures
    fail_tags = []

    sweep_num = sweep["sweep_number"]
    stim_code = sweep["stimulus_code"]
    unit = sweep["stimulus_units"]
    if unit not in data_set.current_clamp_units:
        return {}

    # TODO: verify expected clamp modes
    # if stim in jin["voltage_clamp_stimuli"]:
    #     if unit not in [ "Volts", "mV" ]:
    #        msg = "%s (%s) in wrong mode -- expected voltage clamp" % (name, stim)
    #        fail_tags.append(msg)
    #elif stim_short in jin["current_clamp_stimuli"]:
    #    if unit not in [ "Amps", "pA" ]:
    #        msg = "%s (%s) in wrong mode -- expected current clamp" % (name, stim)
    #        fail_tags.append(msg)

    # pull data streams from file (this is for detecting truncated sweeps)


    if sweep["pre_noise_rms_mv"] > qc_criteria["pre_noise_rms_mv_max"]:
        fail_tags.append("pre-noise: %.3f exceeded qc threshold: %.3f" %(sweep["pre_noise_rms_mv"],qc_criteria["pre_noise_rms_mv_max"]))

    # check Vm and noise at end of recording
    # only do so if acquisition not truncated
    # do not check for ramps, because they do not have
    #   enough time to recover
#    is_ramp = data_set.ontology.stimulus_has_any_tags(sweep["stimulus_code"], data_set.ramp_names)
    is_ramp = sweep['stimulus_name'] in data_set.ramp_names

#    if sweep["completed"]:
    if is_ramp:
        logging.info("sweep %d skipping vrest criteria on ramp", sweep_num)
    else:
        if sweep["post_noise_rms_mv"] > qc_criteria["post_noise_rms_mv_max"]:
            fail_tags.append("post-noise: %.3f exceeded qc threshold: %.3f" % (sweep["post_noise_rms_mv"],qc_criteria["post_noise_rms_mv_max"]))
#   else:
#        fail_tags.append("truncated sweep")

    if sweep["slow_noise_rms_mv"] > qc_criteria["slow_noise_rms_mv_max"]:
        fail_tags.append("slow noise: %.3f above threshold: %.3f" % (sweep["slow_noise_rms_mv"], qc_criteria["slow_noise_rms_mv_max"]) )

    if sweep["vm_delta_mv"] is not None and sweep["vm_delta_mv"] > qc_criteria["vm_delta_mv_max"]:
        fail_tags.append("Vm delta: %.3f above threshold:%.3f" % (sweep["vm_delta_mv"], qc_criteria["vm_delta_mv_max"]))

    # fail sweeps if stimulus duration is zero
    # Uncomment out hte following 3 lines to have sweeps without stimulus
    #   faile QC
    if sweep["stimulus_duration"] <= 0 and not data_set.ontology.stimulus_has_any_tags(stim_code, data_set.extp_names):
        fail_tags.append("No stimulus detected")

    return len(fail_tags) > 0, fail_tags


def evaluate_blowout(blowout_mv, blowout_mv_min, blowout_mv_max, fail_tags):
    if blowout_mv is None or np.isnan(blowout_mv):
        fail_tags.append("Missing blowout value (%s)" % str(blowout_mv))
        return True

    if blowout_mv < blowout_mv_min or blowout_mv > blowout_mv_max:
        fail_tags.append("blowout outside of range")
        return True

    return False


def evaluate_electrode_0(electrode_0_pa, electrode_0_pa_max, fail_tags):
    if electrode_0_pa is None or np.isnan(electrode_0_pa):
        fail_tags.append("electrode_0_pa missing value")
        return True

    if abs(electrode_0_pa) > electrode_0_pa_max:
        fail_tags.append("electrode_0_pa %f exceeds max %f" % (electrode_0_pa, electrode_0_pa_max))
        return True

    return False

def evaluate_seal(seal_gohm, seal_gohm_min, fail_tags):
    if seal_gohm is None or np.isnan(seal_gohm):
        fail_tags.append("Invalid seal (%s)" % str(seal_gohm))
        return True

    if seal_gohm < seal_gohm_min:
        fail_tags.append("seal %f below min %f" % (seal_gohm, seal_gohm_min))
        return True

    return False

def evaluate_input_and_access_resistance(input_access_resistance_ratio,
                                         input_vs_access_resistance_max,
                                         initial_access_resistance_mohm,
                                         access_resistance_mohm_min,
                                         access_resistance_mohm_max,
                                         fail_tags):

    failed_bad_rs = False

    sr_fail_tags = []
    if input_access_resistance_ratio is None:
        failed_bad_rs = True
        sr_fail_tags.append("Resistance ratio not available")

    if initial_access_resistance_mohm is None:
        failed_bad_rs = True
        sr_fail_tags.append("Initial access resistance not available")

    if not failed_bad_rs:
        if initial_access_resistance_mohm < access_resistance_mohm_min:
            failed_bad_rs = True
            sr_fail_tags.append("initial_access_resistance_mohm %f below min %f" % (initial_access_resistance_mohm, access_resistance_mohm_min))
        elif initial_access_resistance_mohm > access_resistance_mohm_max:
            failed_bad_rs = True
            sr_fail_tags.append("initial_access_resistance_mohm %f exceeds max %f" % (initial_access_resistance_mohm, access_resistance_mohm_max))
        if input_access_resistance_ratio > input_vs_access_resistance_max:
            failed_bad_rs = True
            sr_fail_tags.append("input_access_resistance_ratio %f above max %f" % (input_access_resistance_ratio, input_vs_access_resistance_max))

    fail_tags += sr_fail_tags

    return failed_bad_rs

