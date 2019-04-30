import logging
import re

from .ephys_data_set import EphysDataSet


def override_auto_sweep_states(manual_sweep_states,sweep_states):

    for ss in sweep_states:
        for mss in manual_sweep_states:
            if ss["sweep_number"] == mss["sweep_number"]:
                ss["passed"] = mss["passed"]
                if mss["passed"]:
                    ss["reasons"].append("Manually passed")
                else:
                    ss["reasons"].append("Manually failed")


def assign_sweep_states(sweep_states, sweep_features):
    """
    Assign sweep state to all sweeps

    Parameters
    ----------
    sweep_states: dict of sweep states
    sweep_features: list of dics of sweep features

    Returns
    -------

    """
    sweep_states_dict = { s["sweep_number"]:s["passed"] for s in sweep_states }

    for sf in sweep_features:
        sn = sf["sweep_number"]
        if sn in sweep_states_dict:
            sf["passed"] = sweep_states_dict[sn]
        else:
            logging.warning("Could not find QC state for sweep number %d", sn)


def drop_tagged_sweeps(sweep_features):

    sweep_features[:] = [sf for sf in sweep_features if not sf["tags"]]


def drop_failed_sweeps(sweep_features):

    sweep_features[:] = [sf for sf in sweep_features if sf["passed"]]


def remove_sweep_feature(feature_name,sweep_features):

    for sf in sweep_features:
        del sf[feature_name]


def create_sweep_state(sweep_number, fail_tags):

    sweep_state = {'sweep_number': sweep_number,
                   'passed': not fail_tags,
                   'reasons': fail_tags}

    return sweep_state



def extract_sweep_features_subset(feature_names, sweep_features):
    """

    Parameters
    ----------
    sweep_features: list of dicts of sweep features
    feature_names: list of features to select

    Returns
    -------
    sweep_features_subset: list of dicts including only a subset of features from feature_names
    """

    sweep_features_subset = [{k: sf[k] for k in feature_names} for sf in sweep_features]

    return sweep_features_subset


def count_sweep_states(sweep_states):
    """
    Count passed and total sweeps

    Parameters
    ----------
    sweep_states: list of dicts
        Sweep state dict has keys:
            "reason": list of strings
            "sweep_number": int
            "passed": True/False
    Returns
    -------
    num_passed_sweeps: int
        number of sweeps passed QC
    num_sweeps: int
        number of sweeps QCed

    """
    num_passed_sweeps = 0
    for ss in sweep_states:
        if ss["passed"] is True:
            num_passed_sweeps += 1

    num_sweeps = len(sweep_states)

    return num_passed_sweeps, num_sweeps


def modify_sweep_info_keys(sweep_list):
    return [{EphysDataSet.SWEEP_NUMBER: s['sweep_number'],
             EphysDataSet.STIMULUS_UNITS: s['stimulus_units'],
             EphysDataSet.STIMULUS_AMPLITUDE: s['stimulus_absolute_amplitude'],
             EphysDataSet.STIMULUS_CODE: re.sub(r"\[\d+\]", "", s['stimulus_description']),
             EphysDataSet.STIMULUS_NAME: s['stimulus_name'],
             } for s in sweep_list]
