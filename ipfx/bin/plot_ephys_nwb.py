import sys
import matplotlib.pyplot as plt
import numpy as np
from ipfx.dataset.create import create_ephys_data_set
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju


def plot_data_set(data_set, sweep_table, nwb_file_name):

    height_ratios, width_ratios = axes_ratios(sweep_table)

    fig, ax = plt.subplots(len(height_ratios), 3,
                           figsize=(15, 7),
                           gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios}
                           )

    for fig_row, (stimulus_code, sweep_set_table) in enumerate(sweep_table.groupby("stimulus_code")):
        sweep_numbers = sweep_set_table["sweep_number"].sort_values().values
        ss = data_set.sweep_set(sweep_numbers)

        ax_a = ax[fig_row,0]
        ax_i = ax[fig_row,1]
        ax_v = ax[fig_row,2]

        plot_waveforms(ax_i, ss.i, ss.sampling_rate, ss.sweep_number)
        plot_waveforms(ax_v, ss.v, ss.sampling_rate, ss.sweep_number)
        ax_v.get_shared_x_axes().join(ax_i, ax_v)

        clamp_mode = sweep_set_table["clamp_mode"].values[0]
        ax_a.text(0, 0.0, "%s, %s " % (stimulus_code, clamp_mode))
        ax_a.axis('off')

    ax[0,0].set_title("Description")
    ax[0,1].set_title("Current")
    ax[0,2].set_title("Voltage")
    ax[-1,1].set_xlabel("time (s)")
    ax[-1,2].set_xlabel("time (s)")

    fig.suptitle("file: " + nwb_file_name, fontsize=12)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.02,top=0.92)


def axes_ratios(sweep_table):

    height_ratios = []
    width_ratios = [1,3,3]

    for _, sweep_set_table in sweep_table.groupby("stimulus_code"):
        height_ratios.append(len(sweep_set_table.index))

    return height_ratios, width_ratios


def plot_waveforms(ax, ys, rs, sn):

    offset = 0

    for y, r, sn in zip(ys, rs, sn):

        offset += get_vertical_offset(y)
        y += offset
        x = np.arange(0, len(y)) / r

        ax.plot(x, y)
        ax.text(x[0] - 0.05 * (x[-1] - x[0]), y[0], str(sn), fontsize=8)
        ax.set_xlim(x[0], x[-1])

    customize_axis(ax)


def customize_axis(ax):

    ax.tick_params(axis="x", direction="in", pad=-10,labelsize=8)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def get_vertical_offset(data):
    data = data[~np.isnan(data)]
    return np.max(np.abs(data)) * 1.2


def main():

    """
    Plot sweeps of a given ephys nwb file
    # Usage:
    $ python plot_ephys_nwb_file.py NWB_FILE_NAME

    """

    nwb_file = sys.argv[1]
    print("plotting file: %s" % nwb_file)

    stimulus_ontology_file = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
    ont = StimulusOntology(ju.read(stimulus_ontology_file))

    data_set = create_ephys_data_set(nwb_file=nwb_file)
    vclamp_sweep_table = data_set.filtered_sweep_table(clamp_mode=data_set.VOLTAGE_CLAMP)
    plot_data_set(data_set, vclamp_sweep_table, nwb_file)

    data_set = create_ephys_data_set(nwb_file=nwb_file)
    iclamp_sweep_table = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP)
    plot_data_set(data_set, iclamp_sweep_table, nwb_file)

    plt.show()


if __name__ == "__main__": main()
