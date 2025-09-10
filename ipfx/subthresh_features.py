import numpy as np
import logging
from . import time_series_utils as tsu
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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


def voltage_deflection(t, v, i, start, end, deflect_type=None, reject_transients=False, smoothing=True):
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

    if smoothing:
        v_smooth = savgol_filter(v, 40, 2)
        v_window = v_smooth[start_index:end_index]
    else:
        v_window = v[start_index:end_index]

    deflect_index = deflect_func(v_window) + start_index

    # Try to automatically detect and reject transients if requested
    # - look near the peak for high dv/dt values and block them out
    if reject_transients:
        nan_deflect_dispatch = {
            "min": np.nanargmin,
            "max": np.nanargmax,
        }
        nan_deflect_func = nan_deflect_dispatch[deflect_type]

        # use an adaptive threshold to avoid treating the start of the step as a transient
        dvdt_thresh = 1.0 # mV/ms
        t_envelope = 1e3 * (t[start_index:end_index] - t[start_index])
        dvdt_thresh_envelope = 5 * np.exp(-t_envelope / 2) + dvdt_thresh

        window_width = 400
        dvdt = savgol_filter(v, 50, 2, deriv=1, delta=1e3 * (t[1] - t[0])) # mV/ms, smoothed
        if smoothing:
            temp_v = v_smooth.copy()
        else:
            temp_v = v.copy()

        window_start = max(deflect_index - window_width, start_index)
        window_end = min(deflect_index + window_width, end_index)
        dvdt_window = dvdt[window_start:window_end]
        peak_dvdt_ind = np.argmax(np.abs(dvdt_window))
        peak_dvdt = dvdt_window[peak_dvdt_ind]
        peak_dvdt_ind += window_start

        iter_count = 0
        max_iter = 500
        while np.abs(peak_dvdt) > dvdt_thresh_envelope[peak_dvdt_ind - start_index]:
            # found a peak to reject
            iter_count += 1
            if iter_count > max_iter:
                break
            # determine extent of transient

            # find start
            search_start = min(deflect_index, peak_dvdt_ind)
            transient_start_index = np.flatnonzero(np.abs(dvdt[search_start:start_index - 1:-1]) < dvdt_thresh_envelope[search_start - start_index::-1] / 5)[0]
            transient_start_index = search_start - transient_start_index
            print("transient_start_index", transient_start_index)

            transient_base_avg = np.mean(v[transient_start_index - window_width * 2:transient_start_index])
            transient_base_range = 3 * np.std(v[transient_start_index - window_width:transient_start_index])

            # find end
            search_start = max(deflect_index, peak_dvdt_ind)
            print(search_start, transient_base_avg, transient_base_range)
            baseline_return = np.flatnonzero(np.abs(v[search_start:] - transient_base_avg) < transient_base_range)
            if len(baseline_return) > 0:
                transient_end_index = baseline_return[0]
                transient_end_index += search_start
            else:
                transient_end_index = len(t) - 1

            # blank out the transient
            print("blanking", transient_start_index, transient_end_index)

            temp_v[transient_start_index:transient_end_index + 1] = np.nan
            dvdt[transient_start_index:transient_end_index + 1] = np.nan

            # find a new peak
            v_window = temp_v[start_index:end_index]
            deflect_index = nan_deflect_func(v_window) + start_index

            # check the dv/dt around the new peak
            window_start = max(deflect_index - window_width, start_index)
            window_end = min(deflect_index + window_width, end_index)
            dvdt_window = dvdt[window_start:window_end]
            peak_dvdt_ind = np.nanargmax(np.abs(dvdt_window))
            peak_dvdt = dvdt_window[peak_dvdt_ind]
            peak_dvdt_ind += window_start


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

    # Check if this is being done on a hyperpolarizing step
    check_index = tsu.find_time_index(t, end - baseline_interval)
    stim_amp = i[check_index]
    if stim_amp < 0:
        reject_transients = True
    else:
        reject_transients = False

    v_peak, peak_index = voltage_deflection(
        t, v, i, start, end, "min", reject_transients=reject_transients)
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

    # Check if actually hyperpolarizing (otherwise don't use reject_transients option
    # for voltage deflection calculation, since there may be APs)
    check_index = tsu.find_time_index(t, end - baseline_interval)
    stim_amp = i[check_index]
    if stim_amp < 0:
        reject_transients = True
    else:
        reject_transients = False

    v_peak, peak_index = voltage_deflection(t, v, i,
        start, end - baseline_interval, "min", reject_transients=reject_transients)
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

    if type(start) is list:
        starts = start
        ends = end
    else:
        starts = [start] * len(t_set)
        ends = [end] * len(t_set)

    for t, i, v, start, end in zip(t_set, i_set, v_set, starts, end):
        v_peak, min_index = voltage_deflection(
            t, v, i, start, end, 'min', reject_transients=True)
        v_baseline = baseline_voltage(t, v, start, baseline_interval=baseline_interval)
        v_vals.append(v_peak - v_baseline)
        i_vals.append(i[min_index])

    v = np.array(v_vals)
    i = np.array(i_vals)

    if len(v) == 1:
        # If there's just one sweep, we'll have to use its own baseline to estimate
        # the input resistance
        v = np.append(v, 0.)
        i = np.append(i, 0.)

    A = np.vstack([i, np.ones_like(i)]).T
    m, c = np.linalg.lstsq(A, v, rcond=None)[0]

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




