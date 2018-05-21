import os, json
from . import ephys_features as ft
import logging
import numpy as np

DEFAULT_QC_CRITERIA_FILE = os.path.join(os.path.dirname(__file__), 'qc_criteria.json')

def load_default_qc_criteria():
    logging.debug("loading default qc criteria file: %s", DEFAULT_QC_CRITERIA_FILE)
    with open(DEFAULT_QC_CRITERIA_FILE,"r") as f:
        return json.load(f)

def get_last_vm_epoch(idx1, stim, hz):
    return idx1-int(0.500 * hz), idx1

def get_first_vm_noise_epoch(idx0, stim, hz):
    t0 = idx0
    t1 = t0 + int(0.0015 * hz)
    return t0, t1

def get_last_vm_noise_epoch(idx1, stim, hz):
    return idx1-int(0.0015 * hz), idx1

#def get_stability_vm_epoch(idx0, stim, hz):
def get_stability_vm_epoch(idx0, stim_start, hz):
    dur = int(0.500 * hz)
    #stim_start = find_stim_start(idx0, stim)
    if dur > stim_start-1:
        dur = stim_start-1
    elif dur <= 0:
        return 0, 0
    return stim_start-1-dur, stim_start-1

########################################################################
# experiment-level metrics

def measure_blowout(v, idx0):
    return 1e3 * np.mean(v[idx0:])

def measure_electrode_0(curr, hz, t=0.005):
    n_time_steps = int(t * hz)
    # electrode 0 is the average current reading with zero voltage input
    # (ie, the equivalent of resting potential in current-clamp mode)
    return 1e12 * np.mean(curr[0:n_time_steps])

def measure_seal(v, curr, hz):
    t = np.arange(len(v)) / hz
    return 1e-9 * get_r_from_stable_pulse_response(v, curr, t)

# def measure_input_resistance(v, curr, hz):
#     t = np.arange(len(v)) / hz
#     return 1e-6 * get_r_from_stable_pulse_response(v, curr, t)

def measure_input_resistance(v, curr, t):
#    t = np.arange(len(v)) / hz
    return 1e-6 * get_r_from_stable_pulse_response(v, curr, t)


def measure_initial_access_resistance(v, curr, hz):
    t = np.arange(len(v)) / hz
    return 1e-6 * get_r_from_peak_pulse_response(v, curr, t)

def measure_vm(vals):
    if len(vals) < 1:
        return 0, 0
    mean = np.mean(vals)
    rms = np.sqrt(np.mean(np.square(vals-mean)))
    return mean, rms
########################################################################


def get_r_from_stable_pulse_response(v, i, t):
    """Compute input resistance from the stable pulse response

    Parameters
    ----------
    v : float membrane voltage
    i : float input current
    t : time (s)

    Returns
    -------
    ir: float input resistance
    """

    dv = np.diff(v)
    up_idx = np.flatnonzero(dv > 0)
    down_idx = np.flatnonzero(dv < 0)

    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)

    r = []
    for ii in range(len(up_idx)):
        # take average v and i one ms before start
        end = up_idx[ii] - 1
        start = end - one_ms

        avg_v_base = np.mean(v[start:end])
        avg_i_base = np.mean(i[start:end])

        # take average v and i one ms before end
        end = down_idx[ii]-1
        start = end - one_ms

        avg_v_steady = np.mean(v[start:end])
        avg_i_steady = np.mean(i[start:end])

        r_instance = (avg_v_steady-avg_v_base) / (avg_i_steady-avg_i_base)

        r.append(r_instance)

    return np.mean(r)


def get_r_from_peak_pulse_response(v, i, t):
    dv = np.diff(v)
    up_idx = np.flatnonzero(dv > 0)
    down_idx = np.flatnonzero(dv < 0)
    dt = t[1] - t[0]
    one_ms = int(0.001 / dt)
    r = []
    for ii in range(len(up_idx)):
        # take average v and i one ms before
        end = up_idx[ii] - 1
        start = end - one_ms
        avg_v_base = np.mean(v[start:end])
        avg_i_base = np.mean(i[start:end])
        # take average v and i one ms before end
        start = up_idx[ii]
        end = down_idx[ii] - 1
        idx = start + np.argmax(i[start:end])
        avg_v_peak = v[idx]
        avg_i_peak = i[idx]
        r_instance = (avg_v_peak-avg_v_base) / (avg_i_peak-avg_i_base)
        r.append(r_instance)

    return np.mean(r)


def get_sweep_number_by_stimulus_names(data_set, stimulus_names):

    sweeps = data_set.filtered_sweep_table(stimuli=stimulus_names).sort_values(by='sweep_number')
    if len(sweeps) > 1:
        logging.warning("Found multiple sweeps for stimulus %s: using largest sweep number" % str(stimulus_names))

    if len(sweeps) == 0:
        raise IndexError

    return sweeps.sweep_number.values[-1]


def cell_qc_features(data_set, manual_values=None):
    """

    Parameters
    ----------
    data_set : MiesDataSet
        dataset
    manual_values : dict
        default (manual) values that can be passed in through input.json.


    Returns
    -------
    output_data : dict
        cell qc features
    tag_list : list
        tag list

    """
    if manual_values is None:
        manual_values = {}

    output_data = {}
    tag_list = []

    # measure blowout voltage
    try:
        blowout_sweep_number = get_sweep_number_by_stimulus_names(data_set, data_set.blowout_names)
        blowout_data = data_set.sweep(blowout_sweep_number)
        blowout_mv = measure_blowout(blowout_data.v*1e-3,
                                     int(blowout_data.expt_start*blowout_data.sampling_rate))
        output_data['blowout_mv'] = blowout_mv
    except IndexError as e:
        msg = "Blowout is not available"
        tag_list.append(msg)
        logging.warning(msg)
        output_data['blowout_mv'] = None


    # measure "electrode 0"
    try:
        bath_sweep_number = get_sweep_number_by_stimulus_names(data_set, data_set.bath_names)
        bath_data = data_set.sweep(bath_sweep_number)
        e0 = measure_electrode_0(bath_data.v*1e-3,
                                 bath_data.sampling_rate)
        output_data['electrode_0_pa'] = e0
    except IndexError as e:
        msg = "Electrode 0 is not available"
        tag_list.append(msg)
        logging.warning(msg)
        output_data['electrode_0_pa'] = None


    # measure clamp seal
    try:
        seal_sweep_number = get_sweep_number_by_stimulus_names(data_set, data_set.seal_names)
        seal_data = data_set.sweep(seal_sweep_number)
        seal_gohm = measure_seal(seal_data.i*1e-12,
                                 seal_data.v*1e-3,
                                 seal_data.sampling_rate)

        # error may arise in computing seal, which falls through to
        #   exception handler. if seal computation didn't fail but
        #   computation generated invalid value, trigger same
        #   exception handler with different error
        if seal_gohm is None or not np.isfinite(seal_gohm):
            raise ft.FeatureError("Could not compute seal")
    except IndexError as e:
        # seal is not available, for whatever reason. log error
        msg = "Seal is not available"
        tag_list.append(msg)
        logging.warning(msg)
        # look for manual seal value and use it if it's available
        seal = manual_keys.get('manual_seal_gohm', None)
        if seal is not None:
            logging.info("using manual seal value: %f" % seal)
            tag_list.append("Seal set using manual value")
    output_data["seal_gohm"] = seal_gohm


    # measure input and series resistance
    # this requires two steps -- finding the breakin sweep, and then
    #   analyzing it
    # if the value is unavailable then check to see if it was set manually
    breakin_data = None
    try:
        breakin_sweep_number = get_sweep_number_by_stimulus_names(data_set, data_set.breakin_names)
        breakin_data = data_set.sweep(breakin_sweep_number)
    except IndexError as e:
        logging.warning("Error reading breakin sweep.")
        tag_list.append("Breakin sweep not found")
        raise

    ir = None   # input resistance
    sr = None   # series resistance
    if breakin_data is not None:
        ###########################
        # input resistance
        try:
            ir = measure_input_resistance(breakin_data.i*1e-12,
                                          breakin_data.v*1e-3,
                                          breakin_data.t)

        except Exception as e:
            logging.warning("Error reading input resistance.")
            raise

        # apply manual value if it's available
        if ir is None:
            tag_list.append("Input resistance is not available")
            ir = manual_values.get('manual_initial_input_mohm', None)
            if ir is not None:
                msg = "Using manual value for input resistance"
                logging.info(msg)
                tag_list.append(msg);
        ###########################
        # initial access resistance
        try:
            sr = measure_initial_access_resistance(breakin_data.i*1e-12,
                                                   breakin_data.v*1e-3,
                                                   breakin_data.sampling_rate)
        except Exception as e:
            logging.warning("Error reading initial access resistance.")
            raise

        # apply manual value if it's available
        if sr is None:
            tag_list.append("Initial access resistance is not available")
            sr = manual_values.get('manual_initial_access_resistance_mohm', None)
            if sr is not None:
                msg = "Using manual initial access resistance"
                logging.info(msg)
                tag_list.append(msg)
    #
    output_data['input_resistance_mohm'] = ir
    output_data["initial_access_resistance_mohm"] = sr

    sr_ratio = None # input access resistance ratio
    if ir is not None and sr is not None:
        sr_ratio = sr / ir
    else:
        logging.warning("could not compute input/access resistance ratio (sr: %s, ir:: %s)", str(sr), str(ir))

    output_data['input_access_resistance_ratio'] = sr_ratio

    return output_data, tag_list

##############################

def sweep_qc_features(data_set):
    """Compute QC features for iclamp sweeps in the dataset

    Parameters
    ----------
    data_set : MiesDataSet
        dataset

    Returns
    -------
    sweep_features : dict
        sweep features

    """
    sweep_features = []
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True)

    cnt = 0
    for sweep_info in iclamp_sweeps.to_dict(orient='records'):
        sweep_num = sweep_info['sweep_number']

        try:
            sweep_data = data_set.sweep(sweep_num)
        except Exception as e:
            logging.warning("Error reading sweep %d" % sweep_num)
            raise

        sweep = {}

        volts = sweep_data.v*1e-3
        current = sweep_data.i*1e-12
        hz = sweep_data.sampling_rate
        idx_start, idx_stop = int(sweep_data.expt_start*sweep_data.sampling_rate), int(sweep_data.expt_end*sweep_data.sampling_rate)

        # measure Vm and noise before stimulus
        idx0, idx1 = get_first_vm_noise_epoch(idx_start, current, hz)
        _, rms0 = measure_vm(1e3 * volts[idx0:idx1])

        sweep["pre_noise_rms_mv"] = float(rms0)

        # measure Vm and noise at end of recording
        # only do so if acquisition not truncated
        # do not check for ramps, because they do not have enough time to recover
        mean1 = None
        sweep_not_truncated = ( idx_stop == len(current) - 1 )

        if sweep_not_truncated and not data_set.ontology.stimulus_has_any_tags(sweep_info['stimulus_code'], data_set.ramp_names):
            idx0, idx1 = get_last_vm_epoch(idx_stop, current, hz)
            mean1, _ = measure_vm(1e3 * volts[idx0:idx1])
            idx0, idx1 = get_last_vm_noise_epoch(idx_stop, current, hz)
            _, rms1 = measure_vm(1e3 * volts[idx0:idx1])
            sweep["post_vm_mv"] = float(mean1)
            sweep["post_noise_rms_mv"] = float(rms1)
        else:
            sweep["post_noise_rms_mv"] = None

        # measure Vm and noise over extended interval, to check stability
        stim_start = ft.find_stim_start(current, idx_start)
        sweep['stimulus_start_time'] = stim_start / sweep_data.sampling_rate

        idx0, idx1 = get_stability_vm_epoch(idx_start, stim_start, hz)
        mean2, rms2 = measure_vm(1e3 * volts[idx0:idx1])

        slow_noise = float(rms2)
        sweep["slow_vm_mv"] = float(mean2)
        sweep["slow_noise_rms_mv"] = float(rms2)

        # for now (mid-feb 15), make vm_mv the same for pre and slow
        mean0 = mean2
        sweep["pre_vm_mv"] = float(mean0)
        if mean1 is not None:
            delta = abs(mean0 - mean1)
            sweep["vm_delta_mv"] = float(delta)
        else:
            # Use None as 'nan' still breaks the ruby strategies
            sweep["vm_delta_mv"] = None

        # compute stimulus duration, amplitude, interal
        stim_amp, stim_dur = ft.find_stim_amplitude_and_duration(idx_start, current, hz)
        stim_int = ft.find_stim_interval(idx_start, current, hz)

        sweep['stimulus_amplitude'] = stim_amp * 1e12
        sweep['stimulus_duration'] = stim_dur
        sweep['stimulus_interval'] = stim_int
        sweep.update(sweep_info)

        sweep_features.append(sweep)

    return sweep_features

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

            #
        if input_access_resistance_ratio > input_vs_access_resistance_max:
            failed_bad_rs = True
            sr_fail_tags.append("input_access_resistance_ratio %f above max %f" % (input_access_resistance_ratio, input_vs_access_resistance_max))

    fail_tags += sr_fail_tags

    return failed_bad_rs


def qc_experiment(data_set, cell_data, sweep_data, qc_criteria=None):

    """

    Parameters
    ----------
    data_set : MiesDataSet object
        dataset
    cell_data : dict
        cell features
    sweep_data: list of dicts
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

    cell_state = qc_cell(data_set, cell_data, sweep_data, qc_criteria)

    sweep_data_index = { sweep['sweep_number']:sweep for sweep in sweep_data }

    sweep_states = []
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True)

    for sweep_num in iclamp_sweeps.sweep_number:
        sweep = sweep_data_index[sweep_num]

        failed, fail_tags = qc_current_clamp_sweep(data_set, sweep, qc_criteria)
        sweep_state = { 'sweep_number': sweep_num, 'passed': not failed, 'reasons': fail_tags }
        sweep_states.append(sweep_state)

    return cell_state, sweep_states


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
        qc_criteria = load_default_qc_criteria()

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
    sweep_data = data_set.sweep(sweep_num)
    volts = sweep_data.v*1e-3
    current = sweep_data.i*1e-12
    hz = sweep_data.sampling_rate
    idx_start, idx_stop = int(sweep_data.expt_start*hz), int(sweep_data.expt_end*hz)

    if sweep["pre_noise_rms_mv"] > qc_criteria["pre_noise_rms_mv_max"]:
        fail_tags.append("pre-noise exceeded qc threshold")

    # check Vm and noise at end of recording
    # only do so if acquisition not truncated
    # do not check for ramps, because they do not have
    #   enough time to recover
    is_ramp = data_set.ontology.stimulus_has_any_tags(sweep["stimulus_code"], data_set.ramp_names)

    if is_ramp:
        logging.info("sweep %d skipping vrest criteria on ramp", sweep_num)
    else:
        # measure post-stimulus noise
        sweep_not_truncated = ( idx_stop == len(current) - 1 )
        if sweep_not_truncated:
            if sweep["post_noise_rms_mv"] > qc_criteria["post_noise_rms_mv_max"]:
                fail_tags.append("post-noise exceeded qc threshold")
        else:
            fail_tags.append("truncated sweep")

    if sweep["slow_noise_rms_mv"] > qc_criteria["slow_noise_rms_mv_max"]:
        fail_tags.append("slow noise above threshold")

    if sweep["vm_delta_mv"] is not None and sweep["vm_delta_mv"] > qc_criteria["vm_delta_mv_max"]:
        fail_tags.append("Vm delta above threshold")

    # fail sweeps if stimulus duration is zero
    # Uncomment out hte following 3 lines to have sweeps without stimulus
    #   faile QC
    if sweep["stimulus_duration"] <= 0 and not data_set.ontology.stimulus_has_any_tags(stim_code, data_set.extp_names):
        fail_tags.append("No stimulus detected")

    return len(fail_tags) > 0, fail_tags


def qc_cell(data_set, cell_data, sweep_data, qc_criteria=None):
    """Evaluate cell state across different types of stimuli

    Parameters
    ----------
    data_set : MiesDataSet
        data set
    cell_data : dict
        cell features
    sweep_data : list of dicts
        sweep features
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
