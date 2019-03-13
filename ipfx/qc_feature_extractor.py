from . import stim_features as stf
from . import epochs as ep
from . import error as er
import logging
import numpy as np
from . import qc_features as qcf


def extract_blowout(data_set, tags):

    """
    Measure blowout voltage

    Parameters
    ----------
    data_set: EphysDataSet
    tags: list
        warning tags

    Returns
    -------
    blowout_mv: float
        blowout voltage in mV
    """
    ontology = data_set.ontology

    try:
        blowout_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.blowout_names)
        blowout_data = data_set.sweep(blowout_sweep_number)
        expt_start_idx, _ = ep.get_experiment_epoch(blowout_data.i, blowout_data.sampling_rate)
        blowout_mv = qcf.measure_blowout(blowout_data.v, expt_start_idx)

    except IndexError as e:
        tags.append("Blowout is not available")
        blowout_mv = None

    return blowout_mv


def extract_electrode_0(data_set, tags):
    """
    Measure electrode zero

    Parameters
    ----------
    data_set: EphysDataSet
    tags: list
        warning tags

    Returns
    -------
    e0: float
    """

    ontology = data_set.ontology

    try:
        bath_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.bath_names)
        bath_data = data_set.sweep(bath_sweep_number)

        e0 = qcf.measure_electrode_0(bath_data.i, bath_data.sampling_rate)

    except IndexError as e:
        tags.append("Electrode 0 is not available")
        e0 = None

    return e0


def extract_clamp_seal(data_set, tags, manual_values=None):
    """

    Parameters
    ----------
    data_set: EphysDataSet
    tags: list
        warning tags
    manual_values

    Returns
    -------

    """

    ontology = data_set.ontology

    try:
        seal_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.seal_names)
        seal_data = data_set.sweep(seal_sweep_number)

        seal_gohm = qcf.measure_seal(seal_data.v,
                                 seal_data.i,
                                 seal_data.t)

        if seal_gohm is None or not np.isfinite(seal_gohm):
            raise er.FeatureError("Could not compute seal")

    except IndexError as e:
        # seal is not available, for whatever reason. log error
        tags.append("Seal is not available")
        # look for manual seal value and use it if it's available
        seal_gohm = manual_values.get('manual_seal_gohm', None)
        if seal_gohm is not None:
            tags.append("Using manual seal value: %f" % seal_gohm)

    return seal_gohm


def extract_input_and_access_resistance(data_set, tags, manual_values=None):
    """
    Measure input and series (access) resistance in two steps:
        1. finding the breakin sweep
        2. and then analyzing it

    if the value is unavailable then check to see if it was set manually


    Parameters
    ----------
    data_set: EphysDataSet
    tags: list
        warning tags

    manual_values: dict
        manual/default values

    Returns
    -------
    ir: float
        input resistance

    sr: float
        access resistance
    """

    ontology = data_set.ontology


    try:
        breakin_sweep_number = data_set.get_sweep_number_by_stimulus_names(ontology.breakin_names)
        breakin_data = data_set.sweep(breakin_sweep_number)
    except IndexError as e:
        tags.append("Breakin sweep not found")
        breakin_data = None

    ir = None  # input resistance
    sr = None  # series resistance

    if breakin_data is not None:
        ir = extract_input_resistance(breakin_data,tags,manual_values)
        sr = extract_initial_access_resistance(breakin_data,tags,manual_values)

    return ir, sr


def extract_input_resistance(breakin_sweep,tags,manual_values):

    try:
        ir = qcf.measure_input_resistance(breakin_sweep.v,
                                          breakin_sweep.i,
                                          breakin_sweep.t)

    except Exception as e:
        logging.warning("Error reading input resistance.")
        raise

    # apply manual value if it's available
    if ir is None:
        tags.append("Input resistance is not available")
        ir = manual_values.get('manual_initial_input_mohm', None)
        if ir is not None:
            tags.append("Using manual value for input resistance")

    return ir


def extract_initial_access_resistance(breakin_sweep,tags,manual_values):

    try:
        sr = qcf.measure_initial_access_resistance(breakin_sweep.v,
                                                   breakin_sweep.i,
                                                   breakin_sweep.t)

    except Exception as e:
        logging.warning("Error reading initial access resistance.")
        raise

    # apply manual value if it's available
    if sr is None:
        tags.append("Initial access resistance is not available")
        sr = manual_values.get('manual_initial_access_resistance_mohm', None)
        if sr is not None:
            tags.append("Using manual initial access resistance")

    return sr


def compute_input_access_resistance_ratio(ir, sr):

    sr_ratio = None # input access resistance ratio
    if ir is not None and sr is not None:
        sr_ratio = sr / ir
    else:
        logging.warning("Could not compute input/access resistance ratio (sr: %s, ir:: %s)", str(sr), str(ir))

    return sr_ratio


def cell_qc_features(data_set, manual_values=None):
    """

    Parameters
    ----------
    data_set : EphysDataSet
        dataset
    manual_values : dict
        default (manual) values that can be passed in through input.json.


    Returns
    -------
    features : dict
        cell qc features
    tags : list
        warning tags

    """
    if manual_values is None:
        manual_values = {}

    features = {}
    tags = []

    features['blowout_mv'] = extract_blowout(data_set, tags)

    features['electrode_0_pa'] = extract_electrode_0(data_set, tags)

    features["seal_gohm"] = extract_clamp_seal(data_set, tags, manual_values)

    ir, sr = extract_input_and_access_resistance(data_set, tags)

    features['input_resistance_mohm'] = ir
    features["initial_access_resistance_mohm"] = sr

    features['input_access_resistance_ratio'] = compute_input_access_resistance_ratio(ir, sr)

    return features, tags

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
    _, sweep_end_idx = ep.get_recording_epoch(voltage)

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

    _, rec_end_ix = ep.get_recording_epoch(v)
    _, expt_end_ix = ep.get_experiment_epoch(i, hz)

    if rec_end_ix >= expt_end_ix:
        return True
    elif (rec_end_ix / hz) > LONG_RESPONSE_DURATION:
        return True
    else:
        return False
