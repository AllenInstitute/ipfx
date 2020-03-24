import logging
import os
import shutil
import numpy as np
import ipfx.stim_features as st
import ipfx.subthresh_features as subf
import ipfx.epochs as ep
import scipy.signal as sg
import scipy.misc

import datetime
import matplotlib.pyplot as plt
import glob
from allensdk.config.manifest import Manifest

import matplotlib
matplotlib.use('agg')


AXIS_Y_RANGE = [ -110, 60 ]

def get_time_string():
    return datetime.datetime.now().strftime("%I:%M%p %B %d, %Y")

def get_spikes(sweep_features, sweep_number):
    return get_features(sweep_features, sweep_number)["spikes"]

def get_features(sweep_features, sweep_number):
    try:
        return sweep_features[int(sweep_number)]
    except KeyError:
        return sweep_features[str(sweep_number)]

def load_sweep(data_set, sweep_number):
    sweep = data_set.sweep(sweep_number)
    dt = sweep.t[1] - sweep.t[0]
    r = ep.get_experiment_epoch(sweep.i, sweep.sampling_rate)

    return (sweep.v, sweep.i, sweep.t, r, dt)


def plot_single_ap_values(data_set, sweep_numbers, lims_features, sweep_features, cell_features, type_name):
    figs = [ plt.figure() for f in range(3+len(sweep_numbers)) ]

    v, i, t, r, dt = load_sweep(data_set, sweep_numbers[0])
    if type_name == "short_square" or type_name == "long_square":
        stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)
    elif type_name == "ramp":
        stim_start, _, _, start_idx, _ = st.get_stim_characteristics(i, t)

    gen_features = ["threshold", "peak", "trough", "fast_trough", "slow_trough"]
    voltage_features = ["threshold_v", "peak_v", "trough_v", "fast_trough_v", "slow_trough_v"]
    time_features = ["threshold_t", "peak_t", "trough_t", "fast_trough_t", "slow_trough_t"]

    for sn in sweep_numbers:
        spikes = get_spikes(sweep_features, sn)

        if (len(spikes) < 1):
            logging.warning("no spikes in sweep %d" % sn)
            continue

        if type_name != "long_square":
            voltages = [spikes[0][f] for f in voltage_features]
            times = [spikes[0][f] for f in time_features]
        else:
            rheo_sn = cell_features["long_squares"]["rheobase_sweep"]["sweep_number"]
            rheo_spike = get_spikes(sweep_features, rheo_sn)[0]
            voltages = [ rheo_spike[f] for f in voltage_features]
            times = [ rheo_spike[f] for f in time_features]

        plt.figure(figs[0].number)
        plt.scatter(range(len(voltages)), voltages, color='gray')
        plt.tight_layout()


        plt.figure(figs[1].number)
        plt.scatter(range(len(times)), times, color='gray')
        plt.tight_layout()

        plt.figure(figs[2].number)
        plt.scatter([0], [spikes[0]['upstroke'] / (-spikes[0]['downstroke'])], color='gray')
        plt.tight_layout()


    plt.figure(figs[0].number)

    yvals = [float(lims_features[k + "_v_" + type_name]) for k in gen_features if lims_features[k + "_v_" + type_name] is not None]
    xvals = range(len(yvals))

    plt.scatter(xvals, yvals, color='blue', marker='_', s=40, zorder=100)
    plt.xticks(xvals, ['thr', 'pk', 'tr', 'ftr', 'str'])
    plt.title(type_name + ": voltages")

    plt.figure(figs[1].number)
    yvals = [float(lims_features[k + "_t_" + type_name]) for k in gen_features if lims_features[k + "_t_" + type_name] is not None]
    xvals = range(len(yvals))
    plt.scatter(xvals, yvals, color='blue', marker='_', s=40, zorder=100)
    plt.xticks(xvals, ['thr', 'pk', 'tr', 'ftr', 'str'])
    plt.title(type_name + ": times")

    plt.figure(figs[2].number)
    if lims_features["upstroke_downstroke_ratio_" + type_name] is not None:
        plt.scatter([0], [float(lims_features["upstroke_downstroke_ratio_" + type_name])], color='blue', marker='_', s=40, zorder=100)
    plt.xticks([])
    plt.title(type_name + ": up/down")

    for index, sn in enumerate(sweep_numbers):
        plt.figure(figs[3 + index].number)

        v, i, t, r, dt = load_sweep(data_set, sn)
        hz = 1./dt
        expt_start_idx, _ = ep.get_experiment_epoch(i,hz)
        t = t - expt_start_idx*dt
        stim_start_shifted = stim_start - expt_start_idx*dt
        plt.plot(t, v, color='black')
        plt.title(str(sn))

        spikes = get_spikes(sweep_features, sn)
        nspikes = len(spikes)

        delta_v = 5.0

        if nspikes:
            if type_name != "long_square" and nspikes:

                voltages = [spikes[0][f] for f in voltage_features]
                times = [spikes[0][f] for f in time_features]
            else:
                rheo_sn = cell_features["long_squares"]["rheobase_sweep"]["sweep_number"]
                rheo_spike = get_spikes(sweep_features, rheo_sn)[0]
                voltages = [rheo_spike[f] for f in voltage_features]
                times = [rheo_spike[f] for f in time_features]

            plt.scatter(times, voltages, color='red', zorder=20)

            plt.plot([spikes[0]['upstroke_t'] - 1e-3 * (delta_v / spikes[0]['upstroke']),
                      spikes[0]['upstroke_t'] + 1e-3 * (delta_v / spikes[0]['upstroke'])],
                     [spikes[0]['upstroke_v'] - delta_v, spikes[0]['upstroke_v'] + delta_v], color='red')

            if 'downstroke_t' in spikes[0]:
                plt.plot([spikes[0]['downstroke_t'] - 1e-3 * (delta_v / spikes[0]['downstroke']),
                          spikes[0]['downstroke_t'] + 1e-3 * (delta_v / spikes[0]['downstroke'])],
                         [spikes[0]['downstroke_v'] - delta_v, spikes[0]['downstroke_v'] + delta_v], color='red')

        if type_name == "ramp":
            if nspikes:
                plt.xlim(spikes[0]["threshold_t"] - 0.002, spikes[0]["fast_trough_t"] + 0.01)
        elif type_name == "short_square":
            plt.xlim(stim_start_shifted - 0.002, stim_start_shifted + stim_dur + 0.01)
        elif type_name == "long_square":
            plt.xlim(times[0]- 0.002, times[-2] + 0.002)

        plt.tight_layout()


    return figs


def plot_sweep_figures(data_set, image_dir, sizes):

    iclamp_sweep_numbers = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP)['sweep_number'].values
    iclamp_sweep_numbers.sort()
    image_file_sets = {}

    b, a = sg.bessel(4, 0.1, "low")

    for i, sweep_number in enumerate(iclamp_sweep_numbers):

        logging.info("plotting sweep %d" % sweep_number)

        if i == 0:
            v_init, i_init, t_init, r_init, dt_init = load_sweep(data_set, sweep_number)

            if r_init[0] <= 0:
                r_init = (5000,r_init[1])

            tp_steps = int(0.1/dt_init)
            tp_fig = plt.figure()
            axTP = plt.gca()
            axTP.set_title(str(sweep_number))
            axTP.set_ylabel('')
            xTP = t_init[0:tp_steps]
            yTP = v_init[0:tp_steps]
            axTP.plot(xTP, yTP, linewidth=1)
            axTP.set_xlim(xTP[0], xTP[-1])
#            sns.despine()

            exp_fig = plt.figure()
            axDP = plt.gca()
            axDP.set_title(str(sweep_number))
            axDP.set_ylabel('')
            v_exp = v_init[r_init[0]:]
            t_exp = t_init[r_init[0]:]
            yDP = sg.filtfilt(b, a, v_exp, axis=0)
            xDP = t_exp
            baseline = yDP[5000:9000]
            baselineMean = np.mean(baseline)
            baselineV = (np.ones(len(xDP))) * baselineMean
            axDP.plot(xDP, yDP, linewidth=1)
            axDP.plot(xDP, baselineV, linewidth=1)
            axDP.set_xlim(t_exp[0], t_exp[-1])
#            sns.despine()

            v_prev, i_prev, t_prev, r_prev = v_init, i_init, t_init, r_init

        else:
            v, i, t, r, dt = load_sweep(data_set, sweep_number)

            if r[0] <= 0:       # This happens when stimulus starts less than 0.5 s after test pulse
                r = (5000,r[1]) # Manually set start of experiment to a default positive value

            tp_steps = int(0.1/dt_init)

            tp_fig = plt.figure()
            axTP = plt.gca()
#            axTP.set_yticklabels([])
#            axTP.set_xticklabels([])
            axTP.set_title(str(sweep_number))
            axTP.set_ylabel('')
            yTP = v[:tp_steps]
            xTP = t[:tp_steps]
            TPBL = np.mean(yTP[0:100])
            yTPN = yTP - TPBL
            yTPp = v_prev[:tp_steps]
            TPpBL = np.mean(yTPp[0:100])
            yTPpN = yTPp - TPpBL
            yTPi = v_init[:tp_steps]
            TPiBL = np.mean(yTPi[0:100])
            yTPiN = yTPi - TPiBL

            axTP.plot(xTP, yTPiN, linewidth=1, label="init")
            axTP.plot(xTP, yTPpN, linewidth=1, label="prev")
            axTP.plot(xTP, yTPN, linewidth=1, label="this")
            axTP.set_xlim(xTP[0], xTP[-1])

            exp_fig = plt.figure()
            axDP = plt.gca()
#            axDP.set_yticklabels([])
#            axDP.set_xticklabels([])
            axDP.set_title(str(sweep_number))
            axDP.set_ylabel('')
            v_exp = v[r[0]:]
            t_exp = t[r[0]:]
            yDP = sg.filtfilt(b, a, v_exp, axis=0)
            xDP = t_exp
            baseline = yDP[5000:9000]
            baselineMean = np.mean(baseline)
            baselineV = (np.ones(len(xDP))) * baselineMean
            axDP.plot(xDP, yDP, linewidth=1, label="response")
            axDP.plot(xDP, baselineV, linewidth=1, label="baseline")
            axDP.set_xlim(t_exp[0], t_exp[-1])
#            sns.despine()

            if sweep_number == iclamp_sweep_numbers[-1]:
                axTP.legend()
                axDP.legend()

            v_prev, i_prev, t_prev, r_prev = v, i, t, r

        prev_sweep_number = sweep_number

        save_figure(tp_fig, 'test_pulse_%d' % sweep_number, 'test_pulses', image_dir, sizes, image_file_sets)
        save_figure(exp_fig, 'experiment_%d' % sweep_number, 'experiments', image_dir, sizes, image_file_sets)

    return image_file_sets


def save_figure(fig, image_name, image_set_name, image_dir, sizes, image_sets, scalew=1, scaleh=1, ext='png'):
    plt.figure(fig.number)

    if image_set_name not in image_sets:
        image_sets[image_set_name] = { size_name: [] for size_name in sizes }

    for size_name, size in sizes.items():
        fig.set_size_inches(size*scalew, size*scaleh)

        image_file = os.path.join(image_dir, "%s_%s.%s" % (image_name, size_name, ext))
        plt.savefig(image_file)

        image_sets[image_set_name][size_name].append(image_file)

    plt.close()


def plot_images(image_dir, sizes, image_sets):
    image_set_name = "images"
    image_sets[image_set_name] = { size_name: [] for size_name in sizes }

    paths = glob.glob(image_dir + "/*.tif")
    for i, path in enumerate(paths):
        image_data = plt.imread(path)
        image_data = np.array(image_data, dtype=np.float32)

        vmin = image_data.min()
        vmax = image_data.max()

        image_data = np.array((image_data - vmin) / (vmax - vmin) * 255.0, dtype=np.uint8)

        for size_name, size in sizes.items():
            if size:
                s = image_data.shape
                skip = s[0] // size
                sdata = image_data[::skip, ::skip]
            else:
                sdata = image_data


            filename = os.path.join(image_dir, "image_%d_%s.jpg" % (i, size_name))

            scipy.misc.imsave(filename, sdata)
            image_sets['images'][size_name].append(filename)


def plot_subthreshold_long_square_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    lsq_sweeps = cell_features["long_squares"]["sweeps"]
    sub_sweeps = cell_features["long_squares"]["subthreshold_sweeps"]
    tau_sweeps = cell_features["long_squares"]["subthreshold_membrane_property_sweeps"]

    # 0a - Plot VI curve and linear fit, along with vrest
    x = np.array([ s['stim_amp'] for s in sub_sweeps ])
    y = np.array([ s['peak_deflect'][0] for s in sub_sweeps ])
    i = np.array([ s['stim_amp'] for s in tau_sweeps ])

    fig = plt.figure()
    plt.scatter(x, y, color='black')
    plt.plot([x.min(), x.max()], [lims_features["vrest"], lims_features["vrest"]], color="blue", linewidth=2)
    plt.plot(i, i * 1e-3 * lims_features["ri"] + lims_features["vrest"], color="red", linewidth=2)
    plt.xlabel("pA")
    plt.ylabel("mV")
    plt.title("ri = {:.1f}, vrest = {:.1f}".format(lims_features["ri"], lims_features["vrest"]))
    plt.tight_layout()

    save_figure(fig, 'VI_curve', 'subthreshold_long_squares', image_dir, sizes, cell_image_files)

    # 0b - Plot tau curve and average
    fig = plt.figure()
    x = np.array([ s['stim_amp'] for s in tau_sweeps ])
    y = np.array([ s['tau'] for s in tau_sweeps ])
    plt.scatter(x, y, color='black')
    i = np.array([ s['stim_amp'] for s in tau_sweeps ])
    plt.plot([i.min(), i.max()], [cell_features["long_squares"]["tau"], cell_features["long_squares"]["tau"]], color="red", linewidth=2)
    plt.xlabel("pA")
    ylim = plt.ylim()
    plt.ylim(0, ylim[1])
    plt.ylabel("tau (s)")
    plt.tight_layout()


    save_figure(fig, 'tau_curve', 'subthreshold_long_squares', image_dir, sizes, cell_image_files)

    subthresh_dict = {s['sweep_number']:s for s in tau_sweeps}

    # 0c - Plot the subthreshold squares
    tau_sweeps = [ s['sweep_number'] for s in tau_sweeps ]
    tau_figs = [ plt.figure() for i in range(len(tau_sweeps)) ]

    for index, s in enumerate(tau_sweeps):
        v, i, t, r, dt = load_sweep(data_set, s)

        plt.figure(tau_figs[index].number)

        plt.plot(t, v, color="black")

        if index == 0:
            min_y, max_y = plt.ylim()
        else:
            ylims = plt.ylim()
            if min_y > ylims[0]:
                min_y = ylims[0]
            if max_y < ylims[1]:
                max_y = ylims[1]

        stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)
        plt.xlim(stim_start - 0.05, stim_start + stim_dur + 0.05)
        peak_idx = subthresh_dict[s]['peak_deflect'][1]
        peak_t = t[peak_idx]
        plt.scatter([peak_t], [subthresh_dict[s]['peak_deflect'][0]], color='red', zorder=10)
        popt = subf.fit_membrane_time_constant(t, v, stim_start, peak_t)
        plt.title(str(s))
        plt.plot(t[start_idx:peak_idx], exp_curve(t[start_idx:peak_idx] - t[start_idx], *popt), color='blue')
        plt.xlabel("s")


    for index, s in enumerate(tau_sweeps):
        plt.figure(tau_figs[index].number)
        plt.ylim(min_y, max_y)
        plt.tight_layout()

    for index, tau_fig in enumerate(tau_figs):
        save_figure(tau_figs[index], 'tau_%d' % index, 'subthreshold_long_squares', image_dir, sizes, cell_image_files)

def plot_short_square_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    repeat_amp = cell_features["short_squares"].get("stimulus_amplitude", None)

    if repeat_amp is not None:
        short_square_sweep_nums = [ s['sweep_number'] for s in cell_features["short_squares"]["common_amp_sweeps"] ]

        figs = plot_single_ap_values(data_set, short_square_sweep_nums,
                                     lims_features, sweep_features, cell_features,
                                     "short_square")

        for index, fig in enumerate(figs):
            save_figure(fig, 'short_squares_%d' % index, 'short_squares', image_dir, sizes, cell_image_files)

        fig = plot_instantaneous_threshold_thumbnail(data_set, short_square_sweep_nums,
                                                     cell_features, lims_features, sweep_features)

        save_figure(fig, 'instantaneous_threshold_thumbnail', 'short_squares', image_dir, sizes, cell_image_files)


    else:
        logging.warning("No short square figures to plot.")


def plot_instantaneous_threshold_thumbnail(data_set, sweep_numbers, cell_features, lims_features, sweep_features, color='red'):
    min_sweep_number = None
    for sn in sorted(sweep_numbers):
        spikes = get_spikes(sweep_features, sn)

        if len(spikes) > 0:
            min_sweep_number = sn if min_sweep_number is None else min(min_sweep_number, sn)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    v, i, t, r, dt = load_sweep(data_set, sn)
    stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)

    tstart = stim_start - 0.002
    tend = stim_start + stim_dur + 0.005
    tscale = 0.005

    plt.plot(t, v, linewidth=1, color=color)

    plt.ylim(AXIS_Y_RANGE[0], AXIS_Y_RANGE[1])
    plt.xlim(tstart, tend)

    return fig


def plot_ramp_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):

    ramps_sweeps = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
                                                 stimuli=data_set.ontology.ramp_names)
    ramps_sweeps = np.sort(ramps_sweeps['sweep_number'].values)

    figs = []
    if len(ramps_sweeps) > 0:
        figs = plot_single_ap_values(data_set, ramps_sweeps, lims_features, sweep_features, cell_features, "ramp")

        for index, fig in enumerate(figs):
            save_figure(fig, 'ramps_%d' % index, 'ramps', image_dir, sizes, cell_image_files)


def plot_rheo_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    rheo_sweeps = [ lims_features["rheobase_sweep_num"] ]
    figs = plot_single_ap_values(data_set, rheo_sweeps, lims_features, sweep_features, cell_features, "long_square")

    for index, fig in enumerate(figs):
        save_figure(fig, 'rheo_%d' % index, 'rheo', image_dir, sizes, cell_image_files)


def plot_hero_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    fig = plt.figure()
    v, i, t, r, dt = load_sweep(data_set, int(lims_features["thumbnail_sweep_num"]))
    plt.plot(t, v, color='black')
    stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)
    plt.xlim(stim_start - 0.05, stim_start + stim_dur + 0.05)
    plt.ylim(-110, 50)
    spike_times = [spk['threshold_t'] for spk in get_spikes(sweep_features, lims_features["thumbnail_sweep_num"])]
    isis = np.diff(np.array(spike_times))
    plt.title("thumbnail {:d}, amp = {:.1f}".format(lims_features["thumbnail_sweep_num"], stim_amp))
    plt.tight_layout()

    save_figure(fig, 'thumbnail_0', 'thumbnail', image_dir, sizes, cell_image_files, scalew=2)

    fig = plt.figure()
    plt.plot(range(len(isis)), isis)
    plt.ylabel("ISI (ms)")
    if lims_features.get("adaptation", None) is not None:
        plt.title("adapt = {:.3g}".format(lims_features["adaptation"]))
    else:
        plt.title("adapt = not defined")

    plt.tight_layout()
    save_figure(fig, 'thumbnail_1', 'thumbnail', image_dir, sizes, cell_image_files)

    summary_fig = plot_long_square_summary(data_set, cell_features, lims_features)
    save_figure(summary_fig, 'ephys_summary', 'thumbnail', image_dir, sizes, cell_image_files, scalew=2)


def plot_long_square_summary(data_set, cell_features, lims_features):
    long_square_sweeps = cell_features['long_squares']['sweeps']
    long_square_sweep_numbers = [ int(s['sweep_number']) for s in long_square_sweeps ]

    thumbnail_summary_fig = plot_sweep_set_summary(data_set, int(lims_features['thumbnail_sweep_num']), long_square_sweep_numbers)
    plt.figure(thumbnail_summary_fig.number)

    return thumbnail_summary_fig


def plot_fi_curve_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    fig = plt.figure()
    fi_sorted = sorted(cell_features["long_squares"]["spiking_sweeps"], key=lambda s:s['stim_amp'])
    x = [d['stim_amp'] for d in fi_sorted]
    y = [d['avg_rate'] for d in fi_sorted]
    first_nonzero_idx = np.nonzero(y)[0][0]
    plt.scatter(x, y, color='black')
    plt.plot(x[first_nonzero_idx:], cell_features["long_squares"]["fi_fit_slope"] * (np.array(x[first_nonzero_idx:]) - x[first_nonzero_idx]), color='red',linewidth=2)
    plt.xlabel("pA")
    plt.ylabel("spikes/sec")
    plt.title("slope = {:.3g}".format(lims_features["f_i_curve_slope"]))
    rheo_hero_sweeps = [int(lims_features["rheobase_sweep_num"]), int(lims_features["thumbnail_sweep_num"])]
    rheo_hero_x = []
    for s in rheo_hero_sweeps:
        v, i, t, r, dt = load_sweep(data_set, s)
        stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)
        rheo_hero_x.append(stim_amp)
    rheo_hero_y = [ len(get_spikes(sweep_features, s)) for s in rheo_hero_sweeps ]
    plt.scatter(rheo_hero_x, rheo_hero_y, zorder=20, color="blue")
    plt.tight_layout()

    save_figure(fig, 'fi_curve', 'fi_curve', image_dir, sizes, cell_image_files, scalew=2)


def plot_sag_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files):
    fig = plt.figure()
    for d in cell_features["long_squares"]["subthreshold_sweeps"]:
        if d['peak_deflect'][0] == lims_features["vm_for_sag"]:
            v, i, t, r, dt = load_sweep(data_set, int(d['sweep_number']))
            stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)
            plt.plot(t, v, color='black')
            plt.scatter(d['peak_deflect'][1], d['peak_deflect'][0], color='red', zorder=10)
            #plt.plot([stim_start + stim_dur - 0.1, stim_start + stim_dur], [d['steady'], d['steady']], color='red', zorder=10)
    plt.xlim(stim_start - 0.25, stim_start + stim_dur + 0.25)
    plt.title("sag = {:.3g}".format(lims_features['sag']))
    plt.tight_layout()

    save_figure(fig, 'sag', 'sag', image_dir, sizes, cell_image_files, scalew=2)

def mask_nulls(data):
    data[0, np.equal(data[0,:], None) | np.equal(data[0,:],0)] = np.nan

def plot_sweep_value_figures(sweeps, image_dir, sizes, cell_image_files):
    sweeps = sweeps.sort_values('sweep_number').to_dict(orient='records')

    # plot bridge balance
    data = np.array([ [ s['bridge_balance_mohm'], s['sweep_number'] ] for s in sweeps ]).T
    mask_nulls(data)

    fig = plt.figure()
    plt.title('bridge balance')
    plt.plot(data[1,:], data[0,:], marker='.')

    save_figure(fig, 'bridge_balance', 'sweep_values', image_dir, sizes, cell_image_files, scalew=2)

    # plot pre_vm_mv, no blowout sweep
    data = np.array([ [ s['pre_vm_mv'], s['sweep_number'] ]
                      for s in sweeps
                      if not s['stimulus_code'].startswith('EXTPBLWOUT')]).T
    mask_nulls(data)

    fig = plt.figure()
    plt.title('pre vm')
    plt.plot(data[1,:], data[0,:], marker='.')

    save_figure(fig, 'pre_vm_mv', 'sweep_values', image_dir, sizes, cell_image_files, scalew=2)

    # plot bias current
    data = np.array([ [ s['leak_pa'], s['sweep_number'] ] for s in sweeps ]).T
    mask_nulls(data)

    fig = plt.figure()
    plt.title('leak')
    plt.plot(data[1,:], data[0,:], marker='.')

    save_figure(fig, 'leak', 'sweep_values', image_dir, sizes, cell_image_files, scalew=2)

def plot_cell_figures(data_set, figure_data, image_dir, sizes):

    cell_image_files = {}

    plt.style.use('ggplot')

    cell_features = figure_data['cell_features']
    lims_features = figure_data['cell_record']
    sweep_features = figure_data['sweep_features']

    logging.info("saving sweep feature figures")
    plot_sweep_value_figures(data_set.sweep_table, image_dir, sizes, cell_image_files)

    logging.info("saving tau and vi figs")
    plot_subthreshold_long_square_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving short square figs")
    plot_short_square_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving ramps")
    plot_ramp_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving rheo figs")
    plot_rheo_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving thumbnail figs")
    plot_hero_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving fi curve figs")
    plot_fi_curve_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    logging.info("saving sag figs")
    plot_sag_figures(data_set, cell_features, lims_features, sweep_features, image_dir, sizes, cell_image_files)

    return cell_image_files

def plot_sweep_set_summary(data_set, highlight_sweep_number, sweep_numbers,
                           highlight_color='#0779BE', background_color='#dddddd'):

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    for sn in sweep_numbers:
        v, i, t, r, dt = load_sweep(data_set, sn)
        ax.plot(t, v, linewidth=0.5, color=background_color)

    v, i, t, r, dt = load_sweep(data_set, highlight_sweep_number)
    plt.plot(t, v, linewidth=1, color=highlight_color)
    stim_start, stim_dur, stim_amp, start_idx, end_idx = st.get_stim_characteristics(i, t)

    tstart = stim_start - 0.05
    tend = stim_start + stim_dur + 0.25

    ax.set_ylim(AXIS_Y_RANGE[0], AXIS_Y_RANGE[1])
    ax.set_xlim(tstart, tend)

    return fig

def make_sweep_html(sweep_files, file_name, img_sub_dir):
    html = "<html><body>"
    html += "<p>page created at: %s</p>" % get_time_string()

    html += "<p><a href='index.html' target='_blank'>Cell QC Figures</a></p>"

    html += "<div style='position:absolute;width:50%;left:0;top:40'>"
    if 'test_pulses' in sweep_files:
        for small_img, large_img in zip(sweep_files['test_pulses']['small'],
                                        sweep_files['test_pulses']['large']):
            html += "<a href='./%s/%s' target='_blank'><img src='./%s/%s'></img></a>" % ( img_sub_dir, os.path.basename(large_img),
                                                                                          img_sub_dir, os.path.basename(small_img) )
    html += "</div>"

    html += "<div style='position:absolute;width:50%;right:0;top:40'>"
    if 'experiments' in sweep_files:
        for small_img, large_img in zip(sweep_files['experiments']['small'],
                                        sweep_files['experiments']['large']):
            html += "<a href='./%s/%s' target='_blank'><img src='./%s/%s'></img></a>" % ( img_sub_dir, os.path.basename(large_img),
                                                                                          img_sub_dir, os.path.basename(small_img) )
    html += "</div>"

    html += "</body></html>"

    with open(file_name, 'w') as f:
        f.write(html)

def make_cell_html(image_files, metadata, file_name, img_sub_dir, required_fields=( 'electrode_0_pa',
                                                                       'seal_gohm',
                                                                       'initial_access_resistance_mohm',
                                                                       'input_resistance_mohm' )):

    html = "<html><body>"

    html += "<p>page created at: %s</p>" % get_time_string()

    html += "<p><a href='sweep.html' target='_blank'> Sweep QC Figures </a></p>"

    fields = set(required_fields) | set(metadata.keys())

    html += "<table border='1'>"
    for field in sorted(fields):
        html += "<tr><td>%s</td><td>%s</td></tr>" % (field, metadata.get(field,None))
    html += "</table>"

    for image_file_set_name in image_files:
        html += "<h3>%s</h3>" % image_file_set_name

        image_set_files = image_files[image_file_set_name]

        for small_img, large_img in zip(image_set_files['small'], image_set_files['large']):
            html += "<a href='./%s/%s' target='_blank'><img src='./%s/%s'></img></a>" % ( img_sub_dir, os.path.basename(large_img),
                                                                                          img_sub_dir, os.path.basename(small_img) )
    html += ("</body></html>")

    with open(file_name, 'w') as f:
        f.write(html)


def make_sweep_page(data_set, working_dir):
    sizes = { 'small': 2.0, 'large': 6.0 }
    img_sub_dir = "img"
    image_dir = os.path.join(working_dir,img_sub_dir)
    sweep_files = plot_sweep_figures(data_set, image_dir, sizes)
    sweep_page = os.path.join(working_dir, 'sweep.html')
    make_sweep_html(sweep_files, sweep_page, img_sub_dir)


def make_cell_page(data_set, feature_data, working_dir, save_cell_plots=True):
    img_sub_dir = "img"
    image_dir = os.path.join(working_dir,img_sub_dir)

    if save_cell_plots:
        logging.info("Saving cell images")

        sizes = { 'small': 2.0, 'large': 6.0 }
        cell_files = plot_cell_figures(data_set, feature_data, image_dir, sizes)
    else:
        cell_files = {}

    sizes = { 'small': 200, 'large': None }
    plot_images(image_dir, sizes, cell_files)

    cell_page = os.path.join(working_dir, 'index.html')
    make_cell_html(cell_files, feature_data['cell_record'], cell_page, img_sub_dir)

    return cell_page

def exp_curve(x, a, inv_tau, y0):
    ''' Function used for tau curve fitting '''
    return y0 + a * np.exp(-inv_tau * x)


def display_features(qc_fig_dir, data_set, feature_data):
    """

    Parameters
    ----------
    qc_fig_dir: str
        directory name for storing html pages
    data_set: NWB data set
    feature_data: dict
        cell and sweep features

    Returns
    -------

    """
    if os.path.exists(qc_fig_dir):
        logging.warning("Removing existing qc figures directory: %s", qc_fig_dir)
        shutil.rmtree(qc_fig_dir)

    image_dir = os.path.join(qc_fig_dir,"img")
    Manifest.safe_mkdir(qc_fig_dir)
    Manifest.safe_mkdir(image_dir)

    logging.info("Saving figures")
    make_sweep_page(data_set, qc_fig_dir)
    make_cell_page(data_set, feature_data, qc_fig_dir)

