import logging


def assign_sweep_states(manual_sweep_states, auto_sweep_states, sweep_features):

    # start with auto state assignment
    sweep_states = { s["sweep_number"]:s["passed"] for s in auto_sweep_states }

    # override when manual values are available
    for mss in manual_sweep_states:
        sn = mss["sweep_number"]
        logging.debug("Overriding sweep state for sweep %d from %s to %s", sn, str(sweep_states[sn]), mss["passed"])
        sweep_states[sn] = mss["passed"]

    # assign sweep state to all sweeps
    for sweep in sweep_features:
        sn = sweep["sweep_number"]
        if sn in sweep_states:
            sweep["passed"] = sweep_states[sn]
        else:
            logging.warning("could not find QC state for sweep number %d", sn)


def drop_incomplete_sweeps(sweep_features):

    sweep_features[:] = [s for s in sweep_features if s["completed"]]


def remove_sweep_feature(feature_name,sweep_features):

    for sweep in sweep_features:
        del sweep[feature_name]


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

