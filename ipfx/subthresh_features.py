import numpy as np
import logging
from . import time_series_utils as tsu
from scipy.optimize import curve_fit
from . import error as er

def baseline_voltage(t, v, start, baseline_interval=0.1, baseline_detect_thresh=0.3, filter_frequency=1.0):
    # Look at baseline interval before start if start is defined
    if start is not None:
        return tsu.average_voltage(v, t, start - baseline_interval, start)

    logging.info("computing baseline voltage interval")

    # Otherwise try to find an interval where things are pretty flat
    dv = tsu.calculate_dvdt(v, t, filter_frequency)
    non_flat_points = np.flatnonzero(np.abs(dv >= baseline_detect_thresh))
    flat_intervals = t[non_flat_points[1:]] - t[non_flat_points[:-1]]
    long_flat_intervals = np.flatnonzero(flat_intervals >= baseline_interval)

    if long_flat_intervals.size > 0:
        interval_index = long_flat_intervals[0] + 1
        baseline_end_time = t[non_flat_points[interval_index]]
        return tsu.average_voltage(v, t, baseline_end_time - baseline_interval,
                               baseline_end_time)
    else:
        logging.info("Could not find sufficiently flat interval for automatic baseline voltage", RuntimeWarning)
        return np.nan


def voltage_deflection(t, v, i, start, end, deflect_type=None):
    """Measure deflection (min or max, between start and end if specified).

    Parameters
    ----------
    deflect_type : measure minimal ('min') or maximal ('max') voltage deflection
        If not specified, it will check to see if the current (i) is positive or negative
        between start and end, then choose 'max' or 'min', respectively
        If the current is not defined, it will default to 'min'.

    Returns
    -------
    deflect_v : peak
    deflect_index : index of peak deflection
    """

    deflect_dispatch = {
        "min": np.argmin,
        "max": np.argmax,
    }

    start_index = tsu.find_time_index(t, start)
    end_index = tsu.find_time_index(t, end)

    if deflect_type is None:
        if i is not None:
            halfway_index = tsu.find_time_index(t, (end - start) / 2. + start)
            if i[halfway_index] >= 0:
                deflect_type = "max"
            else:
                deflect_type = "min"
        else:
            deflect_type = "min"

    deflect_func = deflect_dispatch[deflect_type]

    v_window = v[start_index:end_index]
    deflect_index = deflect_func(v_window) + start_index

    return v[deflect_index], deflect_index


def time_constant(t, v, i, start, end, max_fit_end=None,
                  frac=0.1, baseline_interval=0.1, min_snr=20.):
    """Calculate the membrane time constant by fitting the voltage response with a
    single exponential.

    Parameters
    ----------
    v : numpy array of voltages in mV
    t : numpy array of times in seconds
    start : start of stimulus interval in seconds
    end : end of stimulus interval in seconds
    max_fit_end : maximum end of exponential fit window. If None, end of fit
        window will always be the time of the peak hyperpolarizing deflection. If set,
        end of fit window will be max_fit_end if it is earlier than the time of peak
        deflection (default None)
    frac : fraction of peak deflection (or deflection at `present_fit_end` if used)
        to find to determine start of fit window. (default 0.1)
    baseline_interval : duration before `start` for baseline Vm calculation
    min_snr : minimum signal-to-noise ratio (SNR) to allow calculation of time constant.
        If SNR is too low, np.nan will be returned. (default 20)

    Returns
    -------
    tau : membrane time constant in seconds
    """
    # Assumes this is being done on a hyperpolarizing step
    v_peak, peak_index = voltage_deflection(t, v, i, start, end, "min")
    if max_fit_end is not None:
        max_peak_index = tsu.find_time_index(t, max_fit_end)
        peak_index = min(max_peak_index, peak_index)
        v_peak = v[peak_index]
    v_baseline = baseline_voltage(t, v, start, baseline_interval=baseline_interval)

    start_index = tsu.find_time_index(t, start)

    # Check that SNR is high enough to proceed
    signal = np.abs(v_baseline - v_peak)
    noise_interval_start_index = tsu.find_time_index(t, start - baseline_interval)
    noise = np.std(v[noise_interval_start_index:start_index])
    if noise == 0: # noiseless - likely a deterministic model
        snr = np.inf
    else:
        snr = signal / noise
    if snr < min_snr:
        logging.debug("signal-to-noise ratio too low for time constant estimate ({:g} < {:g})".format(snr, min_snr))
        return np.nan

    search_result = np.flatnonzero(v[start_index:] <= frac * (v_peak - v_baseline) + v_baseline)

    if not search_result.size:
        raise er.FeatureError("could not find interval for time constant estimate")
    fit_start = t[search_result[0] + start_index]
    fit_end = t[peak_index]

    a, inv_tau, y0 = fit_membrane_time_constant(t, v, fit_start, fit_end)

    return 1. / inv_tau


def sag(t, v, i, start, end, peak_width=0.005, baseline_interval=0.03):
    """Calculate the sag in a hyperpolarizing voltage response.

    Parameters
    ----------
    peak_width : window width to get more robust peak estimate in sec (default 0.005)

    Returns
    -------
    sag : fraction that membrane potential relaxes back to baseline
    """
    v_peak, peak_index = voltage_deflection(t, v, i, start, end, "min")
    v_peak_avg = tsu.average_voltage(v, t, start=t[peak_index] - peak_width / 2.,
                                 end=t[peak_index] + peak_width / 2.)
    v_baseline = baseline_voltage(t, v, start, baseline_interval=baseline_interval)
    v_steady = tsu.average_voltage(v, t, start=end - baseline_interval, end=end)
    sag = (v_peak_avg - v_steady) / (v_peak_avg - v_baseline)

    return sag


def input_resistance(t_set, i_set, v_set, start, end, baseline_interval=0.1):
    """Estimate input resistance in MOhms, assuming all sweeps in passed extractor
    are hyperpolarizing responses."""

    v_vals = []
    i_vals = []
    for t, i, v, in zip(t_set, i_set, v_set):
        v_peak, min_index = voltage_deflection(t, v, i, start, end, 'min')
        v_vals.append(v_peak)
        i_vals.append(i[min_index])

    v = np.array(v_vals)
    i = np.array(i_vals)

    if len(v) == 1:
        # If there's just one sweep, we'll have to use its own baseline to estimate
        # the input resistance
        v = np.append(v, baseline_voltage(t_set[0], v_set[0], start, baseline_interval=baseline_interval))
        i = np.append(i, 0.)

    A = np.vstack([i, np.ones_like(i)]).T
    m, c = np.linalg.lstsq(A, v,rcond=None)[0]

    return m * 1e3


def fit_membrane_time_constant(t, v, start, end, rmse_max_tol = 1.0):
    """Fit an exponential to estimate membrane time constant between start and end

    Parameters
    ----------
    v : numpy array of voltages in mV
    t : numpy array of times in seconds
    start : start of time window for exponential fit
    end : end of time window for exponential fit
    rsme_max_tol: minimal acceptable root mean square error (default 1e-4)

    Returns
    -------
    a, inv_tau, y0 : Coefficients of equation y0 + a * exp(-inv_tau * x)

    returns np.nan for values if fit fails
    """

    start_index = tsu.find_time_index(t, start)
    end_index = tsu.find_time_index(t, end)

    guess = (v[start_index] - v[end_index], 50., v[end_index])
    t_window = (t[start_index:end_index] - t[start_index]).astype(np.float64)
    v_window = v[start_index:end_index].astype(np.float64)
    try:
        popt, pcov = curve_fit(_exp_curve, t_window, v_window, p0=guess)
    except RuntimeError:
        logging.info("Curve fit for membrane time constant failed")
        return np.nan, np.nan, np.nan

    pred = _exp_curve(t_window, *popt)

    rmse = np.sqrt(np.mean((pred - v_window)**2))

    if rmse > rmse_max_tol:
        logging.debug("RMSE %f for the Curve fit for membrane time constant exceeded the maximum tolerance of %f" % (rmse,rmse_max_tol))
        return np.nan, np.nan, np.nan

    return popt

def _exp_curve(x, a, inv_tau, y0):
    return y0 + a * np.exp(-inv_tau * x)




