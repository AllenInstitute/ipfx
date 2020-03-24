import numpy as np
import logging
from . import time_series_utils as tsu
from . import error as er

def detect_putative_spikes(v, t, start=None, end=None, filter=10., dv_cutoff=20., dvdt=None):
    """Perform initial detection of spikes and return their indexes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    start : start of time window for spike detection (optional)
    end : end of time window for spike detection (optional)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    putative_spikes : numpy array of preliminary spike indexes
    """

    if not isinstance(v, np.ndarray):
        raise TypeError("v is not an np.ndarray")

    if not isinstance(t, np.ndarray):
        raise TypeError("t is not an np.ndarray")

    if v.shape != t.shape:
        raise er.FeatureError("Voltage and time series do not have the same dimensions")

    if start is None:
        start = t[0]

    if end is None:
        end = t[-1]

    start_index = tsu.find_time_index(t, start)
    end_index = tsu.find_time_index(t, end)
    v_window = v[start_index:end_index + 1]
    t_window = t[start_index:end_index + 1]

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v_window, t_window, filter)
    else:
        dvdt = dvdt[start_index:end_index]

    # Find positive-going crossings of dV/dt cutoff level
    putative_spikes = np.flatnonzero(np.diff(np.greater_equal(dvdt, dv_cutoff).astype(int)) == 1)

    if len(putative_spikes) <= 1:
        # Set back to original index space (not just window)
        return np.array(putative_spikes) + start_index

    # Only keep spike times if dV/dt has dropped all the way to zero between putative spikes
    putative_spikes = [putative_spikes[0]] + [s for i, s in enumerate(putative_spikes[1:])
        if np.any(dvdt[putative_spikes[i]:s] < 0)]

    # Set back to original index space (not just window)
    return np.array(putative_spikes) + start_index


def find_peak_indexes(v, t, spike_indexes, end=None):
    """Find indexes of spike peaks.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    end : end of time window for spike detection (optional)
    """

    if not end:
        end = t[-1]
    end_index = tsu.find_time_index(t, end)

    spks_and_end = np.append(spike_indexes, end_index)
    peak_indexes = [np.argmax(v[spk:next]) + spk for spk, next in
                    zip(spks_and_end[:-1], spks_and_end[1:])]

    return np.array(peak_indexes)


def filter_putative_spikes(v, t, spike_indexes, peak_indexes, min_height=2.,
                           min_peak=-30., filter=10., dvdt=None):
    """Filter out events that are unlikely to be spikes based on:
        * Height (threshold to peak)
        * Absolute peak level

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
    min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    spike_indexes : numpy array of threshold indexes
    peak_indexes : numpy array of peak indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([]), np.array([])

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    diff_mask = [np.any(dvdt[peak_ind:spike_ind] < 0)
                 for peak_ind, spike_ind
                 in zip(peak_indexes[:-1], spike_indexes[1:])]
    peak_indexes = peak_indexes[np.array(diff_mask + [True])]
    spike_indexes = spike_indexes[np.array([True] + diff_mask)]

    peak_level_mask = v[peak_indexes] >= min_peak
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= min_height
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]

    return spike_indexes, peak_indexes


def find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=10., dvdt=None):
    """Find indexes of maximum upstroke of spike.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of preliminary spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    upstroke_indexes : numpy array of upstroke indexes

    """

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    upstroke_indexes = [np.argmax(dvdt[spike:peak]) + spike for spike, peak in
                        zip(spike_indexes, peak_indexes)]

    return np.array(upstroke_indexes)


def refine_threshold_indexes(v, t, upstroke_indexes, thresh_frac=0.05, filter=10., dvdt=None):
    """Refine threshold detection of previously-found spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    upstroke_indexes : numpy array of indexes of spike upstrokes (for threshold target calculation)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    threshold_indexes : numpy array of threshold indexes
    """

    if not upstroke_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    avg_upstroke = dvdt[upstroke_indexes].mean()
    target = avg_upstroke * thresh_frac

    upstrokes_and_start = np.append(np.array([0]), upstroke_indexes)
    threshold_indexes = []
    for upstk, upstk_prev in zip(upstrokes_and_start[1:], upstrokes_and_start[:-1]):
        potential_indexes = np.flatnonzero(dvdt[upstk:upstk_prev:-1] <= target)
        if not potential_indexes.size:
            # couldn't find a matching value for threshold,
            # so just going to the start of the search interval
            threshold_indexes.append(upstk_prev)
        else:
            threshold_indexes.append(upstk - potential_indexes[0])

    return np.array(threshold_indexes)


def check_thresholds_and_peaks(v, t, spike_indexes, peak_indexes, upstroke_indexes, start=None, end=None,
                               max_interval=0.005, thresh_frac=0.05, filter=10., dvdt=None,
                               tol=1.0, reject_at_stim_start_interval=0.):
    """Validate thresholds and peaks for set of spikes

    Check that peaks and thresholds for consecutive spikes do not overlap
    Spikes with overlapping thresholds and peaks will be merged.

    Check that peaks and thresholds for a given spike are not too far apart.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    upstroke_indexes : numpy array of indexes of spike upstrokes
    start : start of time window for feature analysis (optional)
    end : end of time window for feature analysis (optional)
    max_interval : maximum allowed time between start of spike and time of peak in sec (default 0.005)
    thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)
    tol : tolerance for returning to threshold in mV (optional, default 1)
    reject_at_stim_start_interval : duration of window after start to reject potential spikes (optional, default 0)

    Returns
    -------
    spike_indexes : numpy array of modified spike indexes
    peak_indexes : numpy array of modified spike peak indexes
    upstroke_indexes : numpy array of modified spike upstroke indexes
    clipped : numpy array of clipped status of spikes
    """

    if start is not None and reject_at_stim_start_interval > 0:
        mask = t[spike_indexes] > (start + reject_at_stim_start_interval)
        spike_indexes = spike_indexes[mask]
        peak_indexes = peak_indexes[mask]
        upstroke_indexes = upstroke_indexes[mask]

    overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1] + 1)
    if overlaps.size:
        spike_mask = np.ones_like(spike_indexes, dtype=bool)
        spike_mask[overlaps + 1] = False
        spike_indexes = spike_indexes[spike_mask]

        peak_mask = np.ones_like(peak_indexes, dtype=bool)
        peak_mask[overlaps] = False
        peak_indexes = peak_indexes[peak_mask]

        upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
        upstroke_mask[overlaps] = False
        upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []
    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):
        if t[peak] - t[spk] >= max_interval:
            logging.info("Need to recalculate threshold-peak pair that exceeds maximum allowed interval ({:f} s)".format(max_interval))
            too_long_spikes.append(i)

    if too_long_spikes:
        if dvdt is None:
            dvdt = tsu.calculate_dvdt(v, t, filter)
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * thresh_frac
        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = tsu.find_time_index(t, t[peak] - max_interval)
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            if not below_target.size:
                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]
                t_0 = tsu.find_time_index(t, t[spike] + 2 * max_interval)
                new_peak = np.argmax(v[spike:t_0]) + spike

                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                if t[new_peak] - t[spike] < max_interval and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak
                else:
                    # Otherwise, log and get rid of the spike
                    logging.info("Could not redetermine threshold-peak pair - dropping that pair")
                    drop_spikes.append(i)
#                     raise FeatureError("Could not redetermine threshold")
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]

        if drop_spikes:
            spike_indexes = np.delete(spike_indexes, drop_spikes)
            peak_indexes = np.delete(peak_indexes, drop_spikes)
            upstroke_indexes = np.delete(upstroke_indexes, drop_spikes)

    if not end:
        end = t[-1]
    end_index = tsu.find_time_index(t, end)

    clipped = find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol)

    return spike_indexes, peak_indexes, upstroke_indexes, clipped


def find_clipped_spikes(v, t, spike_indexes, peak_indexes, end_index, tol):
    """
    Check that last spike was not cut off too early by end of stimulus
    by checking that the membrane potential returned to at least the threshold
    voltage - otherwise, drop it

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of indexes of spike peaks
    end_index: int index of the end of time window for feature analysis

    tol: float tolerance to returning to threshold

    Returns
    -------
    clipped: Boolean np.array
    """
    clipped = np.zeros_like(spike_indexes, dtype=bool)

    if len(spike_indexes)>0:
        vtail = v[peak_indexes[-1]:end_index + 1]
        if not np.any(vtail <= v[spike_indexes[-1]] + tol):
            logging.debug(
                "Failed to return to threshold voltage + tolerance (%.2f) after last spike (min %.2f) - marking last spike as clipped",
                v[spike_indexes[-1]] + tol, vtail.min())
            clipped[-1] = True
            logging.debug("max %f, min %f, t(end_index):%f" % (np.max(vtail), np.min(vtail), t[end_index]))

    return clipped

def find_trough_indexes(v, t, spike_indexes, peak_indexes, clipped=None, end=None):
    """
    Find indexes of minimum voltage (trough) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    spike_indexes : numpy array of spike indexes
    peak_indexes : numpy array of spike peak indexes
    end : end of time window (optional)

    Returns
    -------
    trough_indexes : numpy array of threshold indexes
    """

    if not spike_indexes.size or not peak_indexes.size:
        return np.array([])

    if clipped is None:
        clipped = np.zeros_like(spike_indexes, dtype=bool)

    if end is None:
        end = t[-1]
    end_index = tsu.find_time_index(t, end)

    trough_indexes = np.zeros_like(spike_indexes, dtype=float)
    trough_indexes[:-1] = [v[peak:spk].argmin() + peak for peak, spk
                           in zip(peak_indexes[:-1], spike_indexes[1:])]

    if clipped[-1]:
        # If last spike is cut off by the end of the window, trough is undefined
        trough_indexes[-1] = np.nan
    else:
        trough_indexes[-1] = v[peak_indexes[-1]:end_index].argmin() + peak_indexes[-1]

    # nwg - trying to remove this next part for now - can't figure out if this will be needed with new "clipped" method

    # If peak is the same point as the trough, drop that point
#     trough_indexes = trough_indexes[np.where(peak_indexes[:len(trough_indexes)] != trough_indexes)]

    return trough_indexes


def find_downstroke_indexes(v, t, peak_indexes, trough_indexes, clipped=None, filter=10., dvdt=None):
    """Find indexes of minimum voltage (troughs) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of threshold indexes
    clipped: boolean array - False if spike not clipped by edge of window
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    downstroke_indexes : numpy array of downstroke indexes
    """

    if not trough_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes, dtype=bool)

    if len(peak_indexes) < len(trough_indexes):
        raise er.FeatureError("Cannot have more troughs than peaks")
# Taking this out...with clipped info, should always have the same number of points
#     peak_indexes = peak_indexes[:len(trough_indexes)]

    valid_peak_indexes = peak_indexes[~clipped].astype(int)
    valid_trough_indexes = trough_indexes[~clipped].astype(int)

    downstroke_indexes = np.zeros_like(peak_indexes) * np.nan
    downstroke_index_values = [np.argmin(dvdt[peak:trough]) + peak for peak, trough
                         in zip(valid_peak_indexes, valid_trough_indexes)]
    downstroke_indexes[~clipped] = downstroke_index_values

    return downstroke_indexes

