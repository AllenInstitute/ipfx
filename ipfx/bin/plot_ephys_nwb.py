import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
from ipfx.dataset.create import create_ephys_data_set
from ipfx.qc_feature_extractor import sweep_qc_features
from ipfx.utilities import drop_failed_sweeps
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
from typing import (
    Optional, List, Dict, Tuple, Collection, Sequence, Union
)


def plot_data_set(data_set,
            clamp_mode: Optional[str] = None,
            stimuli: Optional[Collection[str]] = None,
            stimuli_exclude: Optional[Collection[str]] = None,
            show_amps: Optional[bool] = True,
            qc_sweeps: Optional[bool] = True,
            figsize=(15, 7),
    ):
    nwb_file_name = str(data_set._data.nwb_file)
    if qc_sweeps:
        drop_failed_sweeps(data_set)
    elif show_amps:
        data_set.sweep_info = sweep_qc_features(data_set)

    sweep_table = data_set.filtered_sweep_table(clamp_mode=clamp_mode, stimuli=stimuli, stimuli_exclude=stimuli_exclude)
    
    if len(sweep_table)==0:
        warnings.warn("No sweeps to plot")
        return
    
    height_ratios, width_ratios = axes_ratios(sweep_table)

    fig, ax = plt.subplots(len(height_ratios), 3,
                           figsize=figsize,
                           gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios}
                           )
    if len(height_ratios)==1:
        # ensure 2d array
        ax = ax[None, :]

    for fig_row, (stimulus_code, sweep_set_table) in enumerate(sweep_table.groupby("stimulus_code")):
        sweep_set_table = sweep_set_table.copy().sort_values("sweep_number", ascending=False)
        sweep_numbers = sweep_set_table["sweep_number"]
        ss = data_set.sweep_set(sweep_numbers)
        if qc_sweeps:
            ss.select_epoch('experiment')
        annot = sweep_numbers.astype(str)
        if show_amps:
            annot += sweep_set_table['stimulus_amplitude'].apply(": {:.3g} pA".format)
            

        ax_a = ax[fig_row,0]
        ax_i = ax[fig_row,1]
        ax_v = ax[fig_row,2]

        plot_waveforms(ax_i, ss.i, ss.sampling_rate, annot)
        plot_waveforms(ax_v, ss.v, ss.sampling_rate)
        ax_v.get_shared_x_axes().join(ax_i, ax_v)

        clamp_mode = sweep_set_table["clamp_mode"].values[0]
        ax_a.text(0, 0.0, "%s \n%s " % (stimulus_code, clamp_mode))
        ax_a.axis('off')

    ax[0,0].set_title("Description")
    ax[0,1].set_title("Current")
    ax[0,2].set_title("Voltage")
    ax[-1,1].set_xlabel("time (s)")
    ax[-1,2].set_xlabel("time (s)")

    fig.suptitle("file: " + nwb_file_name, fontsize=12)

    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'window'):
        mng.resize(*mng.window.maxsize())
    plt.subplots_adjust(left=0.01, right=0.98, bottom=0.02,top=0.92)


def axes_ratios(sweep_table):

    height_ratios = []
    width_ratios = [1,4,4]

    for _, sweep_set_table in sweep_table.groupby("stimulus_code"):
        height_ratios.append(len(sweep_set_table.index))

    return height_ratios, width_ratios


def plot_waveforms(ax, ys, rs, annotations=None):

    offset = 0
    dy = 0
    for i, (y, r) in enumerate(zip(ys, rs)):
        if len(y)==0:
            continue
        y -= y[0]
        dy = max(dy, get_vertical_offset(y))
        offset += dy
        y += offset
        x = np.arange(0, len(y)) / r

        ax.plot(x, y)
        if annotations is not None:
            ax.text(x[0] - 0.01 * (x[-1] - x[0]), y[0], annotations.iloc[i], fontsize=8, ha='right')
    # need to set limits to show all sweeps
    # mode would be best but mean will do fine
    ax.set_xlim(0, np.max([len(y) for y in ys])/r)

    customize_axis(ax)


def customize_axis(ax):

    ax.tick_params(axis="x", direction="in", pad=3, labelsize=8)
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
    plot_data_set(data_set, clamp_mode=data_set.VOLTAGE_CLAMP)
    plot_data_set(data_set, clamp_mode=data_set.CURRENT_CLAMP)

    plt.show()


if __name__ == "__main__": main()
