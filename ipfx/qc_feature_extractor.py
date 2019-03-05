from . import stim_features as stf
from . import epochs as ep
from . import error as er
import logging
import numpy as np
from . import qc_features as qcf


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

        expt_start_idx, _ = ep.get_experiment_epoch(blowout_data.i, blowout_data.sampling_rate)

        blowout_mv = qcf.measure_blowout(blowout_data.v, expt_start_idx)
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

        e0 = qcf.measure_electrode_0(bath_data.i, bath_data.sampling_rate)
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

        seal_gohm = qcf.measure_seal(seal_data.v,
                                 seal_data.i,
                                 seal_data.t)


        # error may arise in computing seal, which falls through to
        #   exception handler. if seal computation didn't fail but
        #   computation generated invalid value, trigger same
        #   exception handler with different error
        if seal_gohm is None or not np.isfinite(seal_gohm):
            raise er.FeatureError("Could not compute seal")
    except IndexError as e:
        # seal is not available, for whatever reason. log error
        msg = "Seal is not available"
        tag_list.append(msg)
        logging.warning(msg)
        # look for manual seal value and use it if it's available
        seal_gohm = manual_values.get('manual_seal_gohm', None)
        if seal_gohm is not None:
            logging.info("using manual seal value: %f" % seal_gohm)
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
            ir = qcf.measure_input_resistance(breakin_data.v,
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
            sr = qcf.measure_initial_access_resistance(breakin_data.v,
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
    data_set : EphysDataSet
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
                                                  exclude_search=True,
                                                  )
    if len(iclamp_sweeps.index)==0:
        logging.warning("No current clamp sweeps available to compute QC features")

    for sweep_info in iclamp_sweeps.to_dict(orient='records'):
        sweep_num = sweep_info['sweep_number']
        sweep_data = data_set.sweep(sweep_num)
        if completed_experiment(sweep_data.i, sweep_data.v,sweep_data.sampling_rate):
            sweep = current_clamp_sweep_qc_features(sweep_data,sweep_info,ontology)
            sweep["completed"] = True
        else:
            sweep = dict()
            sweep["completed"] = False
            logging.info("sweep {}, {}, did not complete experiment".format(sweep_num, sweep_info["stimulus_name"]))
        sweep.update(sweep_info)
        sweep_features.append(sweep)
    return sweep_features


def current_clamp_sweep_qc_features(sweep_data,sweep_info,ontology):

    sweep = {}

    voltage = sweep_data.v
    current = sweep_data.i
    t = sweep_data.t
    hz = sweep_data.sampling_rate

    expt_start_idx, _ = ep.get_experiment_epoch(current, hz)
    _, sweep_end_idx = ep.get_sweep_epoch(voltage)

    # measure Vm and noise before stimulus
    idx0, idx1 = ep.get_first_vm_noise_epoch(expt_start_idx, hz)  # count from the beginning of the experiment

    _, rms0 = qcf.measure_vm(voltage[idx0:idx1])

    sweep["pre_noise_rms_mv"] = float(rms0)

    # measure Vm and noise at end of recording
    # do not check for ramps, because they do not have enough time to recover
    mean_last_vm_epoch = None

    is_ramp = sweep_info['stimulus_name'] in ontology.ramp_names

    if not is_ramp:
        idx0, idx1 = ep.get_last_vm_epoch(sweep_end_idx, hz)
        mean_last_vm_epoch, _ = qcf.measure_vm(voltage[idx0:idx1])
        idx0, idx1 = ep.get_last_vm_noise_epoch(sweep_end_idx, hz)
        _, rms1 = qcf.measure_vm(voltage[idx0:idx1])
        sweep["post_vm_mv"] = float(mean_last_vm_epoch)
        sweep["post_noise_rms_mv"] = float(rms1)
    else:
        sweep["post_noise_rms_mv"] = None

    # measure Vm and noise over extended interval, to check stability

    stim_start_time, stim_dur, stim_amp, stim_start_idx, stim_end_idx = stf.get_stim_characteristics(current,t)

    sweep['stimulus_start_time'] = stim_start_time

    idx0, idx1 = ep.get_stability_vm_epoch(stim_start_idx, hz)
    mean2, rms2 = qcf.measure_vm(voltage[idx0:idx1])

    sweep["slow_vm_mv"] = float(mean2)
    sweep["slow_noise_rms_mv"] = float(rms2)

    mean0 = mean2
    sweep["pre_vm_mv"] = float(mean0)

    if mean_last_vm_epoch is not None:
        delta = abs(mean0 - mean_last_vm_epoch)
        sweep["vm_delta_mv"] = float(delta)
    else:
        # Use None as 'nan' still breaks the ruby strategies
        sweep["vm_delta_mv"] = None

    stim_int = stf.find_stim_interval(expt_start_idx, current, hz)

    sweep['stimulus_amplitude'] = stim_amp
    sweep['stimulus_duration'] = stim_dur
    sweep['stimulus_interval'] = stim_int

    return sweep


def completed_experiment(i,v,hz):

    LONG_RESPONSE_DURATION = 5  # this will count long ramps as completed

    _, sweep_end_ix = ep.get_sweep_epoch(v)
    _, expt_end_ix = ep.get_experiment_epoch(i, hz)

    if sweep_end_ix >= expt_end_ix:
        return True
    elif (sweep_end_ix / hz) > LONG_RESPONSE_DURATION:
        return True
    else:
        return False
