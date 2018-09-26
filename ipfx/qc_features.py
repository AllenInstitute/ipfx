from . import ephys_features as ft
import logging
import numpy as np
import stim_features as st

POST_STIM_STABILITY_INTERVAL = 0.5
LONG_RESPONSE_DURATION = 5  # this will count long ramps as completed


def measure_blowout(v, idx0):

    return 1e3 * np.mean(v[idx0:]*1e-3)


def measure_electrode_0(curr, hz, t=0.005):

    n_time_steps = int(t * hz)
    # electrode 0 is the average current reading with zero voltage input
    # (ie, the equivalent of resting potential in current-clamp mode)
    return 1e12 * np.mean(curr[0:n_time_steps]*1e-12)


def measure_seal(v, curr, t):

    return 1e-9 * get_r_from_stable_pulse_response(v*1e-3, curr*1e-12, t)


def measure_input_resistance(v, curr, t):

    return 1e-6 * get_r_from_stable_pulse_response(v*1e-3, curr*1e-12, t)


def measure_initial_access_resistance(v, curr, t):
    return 1e-6 * get_r_from_peak_pulse_response(v*1e-3, curr*1e-12, t)


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
    v : float membrane voltage (V)
    i : float input current (A)
    t : time (s)

    Returns
    -------
    ir: float input resistance
    """


    dv = np.diff(v)
    up_idx = np.flatnonzero(dv > 0)
    down_idx = np.flatnonzero(dv < 0)
    assert len(up_idx) == len(down_idx), "Truncated breakin sweep, truncated response to a square pulse"
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
    assert len(up_idx) == len(down_idx), "Truncated breakin sweep, truncated response to a square pulse"

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
    ontology = data_set.ontology
    # measure blowout voltage
    try:
        blowout_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.blowout_names)
        blowout_data = data_set.sweep(blowout_sweep_number)
        blowout_mv = measure_blowout(blowout_data.v, blowout_data.expt_idx_range[0])
        output_data['blowout_mv'] = blowout_mv
    except IndexError as e:
        msg = "Blowout is not available"
        tag_list.append(msg)
        logging.warning(msg)
        output_data['blowout_mv'] = None


    # measure "electrode 0"
    try:
        bath_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.bath_names)
        bath_data = data_set.sweep(bath_sweep_number)

        e0 = measure_electrode_0(bath_data.i, bath_data.sampling_rate)
        output_data['electrode_0_pa'] = e0
    except IndexError as e:
        msg = "Electrode 0 is not available"
        tag_list.append(msg)
        logging.warning(msg)
        output_data['electrode_0_pa'] = None


    # measure clamp seal
    try:
        seal_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.seal_names)
        seal_data = data_set.sweep(seal_sweep_number)

        seal_gohm = measure_seal(seal_data.v,
                                 seal_data.i,
                                 seal_data.t)


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
        seal_gohm = manual_values.get('manual_seal_gohm', None)
        if seal_gohm is not None:
            logging.info("using manual seal value: %f" % seal)
            tag_list.append("Seal set using manual value")
    output_data["seal_gohm"] = seal_gohm


    # measure input and series resistance
    # this requires two steps -- finding the breakin sweep, and then
    #   analyzing it
    # if the value is unavailable then check to see if it was set manually
    breakin_data = None
    try:
        breakin_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.breakin_names)
        breakin_data = data_set.sweep(breakin_sweep_number)
    except IndexError as e:
        logging.warning("Error reading breakin sweep.")
        tag_list.append("Breakin sweep not found")
#        raise

    ir = None   # input resistance
    sr = None   # series resistance
    if breakin_data is not None:
        ###########################
        # input resistance
        try:
            ir = measure_input_resistance(breakin_data.v,
                                          breakin_data.i,
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
            sr = measure_initial_access_resistance(breakin_data.v,
                                                   breakin_data.i,
                                                   breakin_data.t)

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
    data_set : AibsDataSet
        dataset

    Returns
    -------
    sweep_features : list of dicts
        each dict includes features of a sweep

    """
    ontology = data_set.ontology
    sweep_features = []
    iclamp_sweeps = data_set.filtered_sweep_table(current_clamp_only=True,
                                                  exclude_test=True,
                                                  exclude_search=True)
    if len(iclamp_sweeps.index)==0:
        raise ValueError("No current clamp sweeps available for QC.")

    for sweep_info in iclamp_sweeps.to_dict(orient='records'):
        sweep_num = sweep_info['sweep_number']
        sweep_data = data_set.sweep(sweep_num)

        sweep = {}

        voltage = sweep_data.v
        current = sweep_data.i
        t = sweep_data.t
        hz = sweep_data.sampling_rate
        expt_start_idx, expt_end_idx = sweep_data.expt_idx_range

        # measure Vm and noise before stimulus
        idx0, idx1 = st.get_first_vm_noise_epoch(expt_start_idx, hz) # count from the beginning of the experiment

        _, rms0 = measure_vm(voltage[idx0:idx1])

        sweep["pre_noise_rms_mv"] = float(rms0)

        # measure Vm and noise at end of recording
        # only do so if acquisition not truncated
        # do not check for ramps, because they do not have enough time to recover
        mean_last_vm_epoch = None

        is_ramp = sweep_info['stimulus_name'] in ontology.ramp_names

        if not is_ramp:
            idx0, idx1 = st.get_last_vm_epoch(expt_end_idx, hz)
            mean_last_vm_epoch, _ = measure_vm(voltage[idx0:idx1])
            idx0, idx1 = st.get_last_vm_noise_epoch(expt_end_idx, hz)
            _, rms1 = measure_vm(voltage[idx0:idx1])
            sweep["post_vm_mv"] = float(mean_last_vm_epoch)
            sweep["post_noise_rms_mv"] = float(rms1)
        else:
            sweep["post_noise_rms_mv"] = None

        # measure Vm and noise over extended interval, to check stability

        stim_start_ix = st.find_stim_start(current, expt_start_idx)
        sweep['stimulus_start_time'] = t[stim_start_ix]
        idx0, idx1 = st.get_stability_vm_epoch(stim_start_ix, hz)
        mean2, rms2 = measure_vm(voltage[idx0:idx1])

        sweep["slow_vm_mv"] = float(mean2)
        sweep["slow_noise_rms_mv"] = float(rms2)

        # for now (mid-feb 15), make vm_mv the same for pre and slow
        mean0 = mean2
        sweep["pre_vm_mv"] = float(mean0)

        if mean_last_vm_epoch is not None:
            delta = abs(mean0 - mean_last_vm_epoch)
            sweep["vm_delta_mv"] = float(delta)
        else:
            # Use None as 'nan' still breaks the ruby strategies
            sweep["vm_delta_mv"] = None

        # compute stimulus duration, amplitude, interval
        stim_amp, stim_dur = st.find_stim_amplitude_and_duration(expt_start_idx, current, hz)
        stim_int = st.find_stim_interval(expt_start_idx, current, hz)

        sweep['stimulus_amplitude'] = stim_amp
        sweep['stimulus_duration'] = stim_dur
        sweep['stimulus_interval'] = stim_int

        sweep.update(sweep_info)

        sweep_features.append(sweep)

    return sweep_features







