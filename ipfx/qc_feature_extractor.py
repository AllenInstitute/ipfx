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
        blowout_sweep_number = data_set.get_sweep_number(ontology.blowout_names)
        blowout_data = data_set.sweep(blowout_sweep_number)
        _,test_end_idx = blowout_data.epochs["test"]
        blowout_mv = qcf.measure_blowout(blowout_data.v, test_end_idx)
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
        bath_sweep_number = data_set.get_sweep_number(ontology.bath_names)
        bath_data = data_set.sweep(bath_sweep_number)

        e0 = qcf.measure_electrode_0(bath_data.i, bath_data.sampling_rate)

    except IndexError as e:
        tags.append("Electrode 0 is not available")
        e0 = None

    return e0


def extract_recording_date(data_set, tags):

    try:
        recording_date = data_set.get_recording_date()

    except KeyError as e:
        tags.append("Recording date is missing")
        recording_date = None

    return recording_date


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
        seal_sweep_number = data_set.get_sweep_number(ontology.seal_names,"VoltageClamp")
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
        breakin_sweep_number = data_set.get_sweep_number(ontology.breakin_names,"VoltageClamp")
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

    features['recording_date'] = extract_recording_date(data_set, tags)

    features["seal_gohm"] = extract_clamp_seal(data_set, tags, manual_values)

    ir, sr = extract_input_and_access_resistance(data_set, tags)

    features['input_resistance_mohm'] = ir
    features["initial_access_resistance_mohm"] = sr

    features['input_access_resistance_ratio'] = compute_input_access_resistance_ratio(ir, sr)

    return features, tags


def sweep_qc_features(data_set):

    ontology = data_set.ontology
    sweeps_features = []
    iclamp_sweeps = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
                                                  stimuli_exclude=["Test", "Search"],
                                                  )
    if len(iclamp_sweeps.index) == 0:
        logging.warning("No current clamp sweeps available to compute QC features")

    for sweep_info in iclamp_sweeps.to_dict(orient='records'):
        sweep_features = {}
        sweep_features.update(sweep_info)

        sweep_num = sweep_info['sweep_number']
        sweep = data_set.sweep(sweep_num)
        is_ramp = sweep_info['stimulus_name'] in ontology.ramp_names
        tags = check_sweep_integrity(sweep, is_ramp)
        sweep_features["tags"] = tags

        stim_features = current_clamp_sweep_stim_features(sweep)
        sweep_features.update(stim_features)

        if not tags:
            qc_features = current_clamp_sweep_qc_features(sweep, is_ramp)
            sweep_features.update(qc_features)
        else:
            logging.warning("sweep {}: {}".format(sweep_num, tags))

        sweeps_features.append(sweep_features)

    return sweeps_features


def check_sweep_integrity(sweep, is_ramp):

    tags = []

    for k,v in sweep.epochs.items():
        if not v:
            tags.append(F"{k} epoch is missing")

    if not is_ramp:
        if sweep.epochs["recording"] and sweep.epochs["experiment"]:
            if sweep.epochs["recording"][1] < sweep.epochs["experiment"][1]:
                tags.append("Recording stopped before completing the experiment epoch")

    return tags


def current_clamp_sweep_stim_features(sweep):

    stim_features = {}

    i = sweep.i
    t = sweep.t
    hz = sweep.sampling_rate
    start_time, dur, amp, start_idx, end_idx = stf.get_stim_characteristics(i, t)

    stim_features['stimulus_start_time'] = start_time
    stim_features['stimulus_amplitude'] = amp
    stim_features['stimulus_duration'] = dur

    if sweep.epochs["experiment"]:
        expt_start_idx, _ = sweep.epochs["experiment"]
        interval = stf.find_stim_interval(expt_start_idx, i, hz)
    else:
        interval = None

    stim_features['stimulus_interval'] = interval

    return stim_features


def current_clamp_sweep_qc_features(sweep, is_ramp):

    qc_features = {}

    voltage = sweep.v
    current = sweep.i
    hz = sweep.sampling_rate

    expt_start_idx, _ = ep.get_experiment_epoch(current, hz)
    # measure noise before stimulus
    idx0, idx1 = ep.get_first_noise_epoch(expt_start_idx, hz)  # count from the beginning of the experiment
    _, qc_features["pre_noise_rms_mv"] = qcf.measure_vm(voltage[idx0:idx1])

    # measure mean and rms of Vm at end of recording
    # do not check for ramps, because they do not have enough time to recover

    _, rec_end_idx = ep.get_recording_epoch(voltage)

    if not is_ramp:
        idx0, idx1 = ep.get_last_stability_epoch(rec_end_idx, hz)
        mean_last_stability_epoch, _ = qcf.measure_vm(voltage[idx0:idx1])

        idx0, idx1 = ep.get_last_noise_epoch(rec_end_idx, hz)
        _, rms_last_noise_epoch = qcf.measure_vm(voltage[idx0:idx1])
    else:
        rms_last_noise_epoch = None
        mean_last_stability_epoch = None

    qc_features["post_vm_mv"] = mean_last_stability_epoch
    qc_features["post_noise_rms_mv"] = rms_last_noise_epoch

    # measure mean and rms of Vm and over extended interval before stimulus, to check stability

    stim_start_idx, _ = ep.get_stim_epoch(current)

    idx0, idx1 = ep.get_first_stability_epoch(stim_start_idx, hz)
    mean_first_stability_epoch, rms_first_stability_epoch = qcf.measure_vm(voltage[idx0:idx1])

    qc_features["pre_vm_mv"] = mean_first_stability_epoch
    qc_features["slow_vm_mv"] = mean_first_stability_epoch
    qc_features["slow_noise_rms_mv"] = rms_first_stability_epoch

    qc_features["vm_delta_mv"] = qcf.measure_vm_delta(mean_first_stability_epoch, mean_last_stability_epoch)

    return qc_features


