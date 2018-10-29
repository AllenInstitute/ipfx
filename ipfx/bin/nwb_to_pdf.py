#!/bin/env python
# vim: set fileencoding=utf-8 :
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.backends.backend_pdf import PdfPages

import json
import numpy as np
import math
import argparse
import os

from pynwb import NWBHDF5IO


def physical(number, unit):
    if math.isnan(number):
        return 'NaN'
    return "%s%s" % (to_si(number), unit)


def to_si(d, sep=' '):
    """
    taken from https://stackoverflow.com/a/15734251/7809404

    Convert number to string with SI prefix

    :Example:

    >>> to_si(2500.0)
    '2.5 k'

    >>> to_si(2.3E6)
    '2.3 M'

    >>> to_si(2.3E-6)
    '2.3 µ'

    >>> to_si(-2500.0)
    '-2.5 k'

    >>> to_si(0)
    '0'

    """

    incPrefixes = ['k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    decPrefixes = ['m', 'µ', 'n', 'p', 'f', 'a', 'z', 'y']

    if d == 0:
        return str(0)

    degree = int(math.floor(math.log10(math.fabs(d)) / 3))

    prefix = ''

    if degree != 0:
        ds = degree/math.fabs(degree)
        if ds == 1:
            if degree - 1 < len(incPrefixes):
                prefix = incPrefixes[degree - 1]
            else:
                prefix = incPrefixes[-1]
                degree = len(incPrefixes)

        elif ds == -1:
            if -degree - 1 < len(decPrefixes):
                prefix = decPrefixes[-degree - 1]
            else:
                prefix = decPrefixes[-1]
                degree = -len(decPrefixes)

        scaled = round(float(d * math.pow(1000, -degree)), 3)

        s = "{scaled}{sep}{prefix}".format(scaled=scaled,
                                           sep=sep,
                                           prefix=prefix)

    else:
        s = "{d}".format(d=round(d, 3))

    return(s)


class SweepCollection():
    '''
    Class for grouping sweep related PatchClampSeries
    '''
    def __init__(self):
        self.data = {}

    def get(self, id):
        if id not in self.data:
            self.data[id] = SingleSweep(id)
        return self.data[id]

    def __iter__(self):
        def _sort(id):
            return sorted(self.get(id).get_acquisition().keys())
        for s in sorted(self.data.keys(), key=_sort):
            yield s


class SingleSweep():
    '''
    Generic Class for storing sweep specific PatchClampSeries Data
    '''
    def __init__(self, id):
        self.id = id
        # initialize associated sweepdata
        self.acquisition = {}
        self.stimulus = {}

    def add_acquisition(self, key, pcs):
        self.acquisition[key] = PatchClampSeriesPlotData(pcs)

    def add_stimulus(self, key, pcs):
        self.stimulus[key] = PatchClampSeriesPlotData(pcs)

    def get_acquisition(self):
        return self.acquisition

    def get_stimulus(self):
        return self.stimulus

    def num_acquisition(self):
        return len(self.acquisition)

    def num_stimulus(self):
        return len(self.stimulus)


class PatchClampSeriesPlotData():
    '''
    Data class for storing plotting information for PatchClampSeries
        pcs: neurodata patchClampSeries or derived object
    '''
    def __init__(self, pcs):
        self.cycle_id = json.loads(pcs.description).get('cycle_id')
        self.type = pcs.neurodata_type
        self.name = pcs.name

        self._load_data(pcs)
        self._annotation(pcs)
        self._title()

    def _load_data(self, pcs):
        self.data = {}
        self.unit = {}
        self.axis = {}

        self.data['y'] = pcs.data[()]
        attributes = pcs.data.attrs

        conv = attributes.get('conversion')
        unit = attributes.get('unit')
        self.unit['y'] = physical(conv, unit).split("1.0").pop()

        if unit == "A":
            self.axis['y'] = 'Current'
        elif unit == "V":
            self.axis['y'] = 'Voltage'
        else:
            self.axis['y'] = ''

        _start = pcs.starting_time
        _step = 1 / pcs.rate
        _len = len(self.data['y'])
        _stop = _start + _step * _len
        self.data['x'] = np.linspace(_start, _stop, num=_len, endpoint=False)
        self.unit['x'] = pcs.time_unit
        if self.unit['x'] == "Seconds":
            self.unit['x'] = "s"
        self.axis['x'] = 'time'

    def _annotation(self, pcs):
        self.annotation = []
        self.add_annotation("desc", pcs.stimulus_description, None)
        self.add_annotation("rate", pcs.rate, "Hz")
        self.add_annotation("gain", pcs.gain, "x")
        if self.check_type('CurrentClampSeries'):
            self.add_annotation("bias current",
                                pcs.bias_current, "A")
            self.add_annotation("bridge balance",
                                pcs.bridge_balance, "Ω")
            self.add_annotation("capacitance compensation",
                                pcs.capacitance_compensation, "F")
        elif self.check_type('VoltageClampSeries'):
            self.add_annotation("capacitance_fast",
                                pcs.capacitance_fast, "F")
            self.add_annotation("capacitance_slow",
                                pcs.capacitance_slow, "F")
            self.add_annotation("resistance_comp_bandwidth",
                                pcs.resistance_comp_bandwidth, "Hz")
            self.add_annotation("resistance_comp_correction",
                                pcs.resistance_comp_correction, "%")
            self.add_annotation("resistance_comp_prediction",
                                pcs.resistance_comp_prediction, "%")
            self.add_annotation("whole_cell_capacitance_comp",
                                pcs.whole_cell_capacitance_comp, "F")
            self.add_annotation("whole_cell_series_resistance_comp",
                                pcs.whole_cell_series_resistance_comp, "Ω")

    def _title(self):
        self.title = "%s: %s" % (self.type, self.name)

    def check_type(self, type_):
        return self.type == type_

    def get_annotation(self):
        return '\n'.join(self.annotation)

    def add_annotation(self, name, data, unit):
        if unit is None:
            self.annotation.append("%s: %s" % (name, data))
        else:
            self.annotation.append("%s: %s" % (name, physical(data, unit)))


def gather_sweeps(nwbfile):
    '''
    sort PatchClampSeries according to cycle_id
    '''
    sweeps = SweepCollection()
    with NWBHDF5IO(nwbfile, 'r') as io:
        nwb = io.read()
        for key in nwb.acquisition:
            acquisition = nwb.get_acquisition(key)
            description = json.loads(acquisition.description)
            cycle_id = description['cycle_id']
            sweeps.get(cycle_id).add_acquisition(key, acquisition)
        for key in nwb.stimulus:
            ccss = nwb.get_stimulus(key)
            description = json.loads(ccss.description)
            cycle_id = description['cycle_id']
            sweeps.get(cycle_id).add_stimulus(key, ccss)
    return sweeps


def plot_patchClampSeries(axis, pcs_data_plot):
    '''
    plot a PatchClampSeries against the axis
        pcs_data_plot: class PatchClampSeriesPlotData
        axis:    plt.axis
    '''
    axis.plot(pcs_data_plot.data['x'], pcs_data_plot.data['y'])
    axis.set_title("%s" % pcs_data_plot.title)
    axis.set_ylabel('%s [%s]' % (pcs_data_plot.axis['y'],
                                 pcs_data_plot.unit['y']))
    axis.xaxis.set_tick_params(labelbottom=False)
    props = {"boxstyle": 'round', "facecolor": 'wheat', "alpha": 0.5}
    axis.text(0.05, 0.95, pcs_data_plot.get_annotation(),
              transform=axis.transAxes,
              fontsize=8,
              verticalalignment='top',
              bbox=props)


def plot_sweepdata(sweepdata, axes, addXTicks=False):
    '''
    plot the given sweep data (stimulus or acquisition) on the given axes
        sweepdata: dict(class PatchClampSeriesPlotData) (either acquisition or
                   stimulus)
        axes:      np.ndarray(plt.axis)
        addXTicks: Add ticks and a label to the X axis at the bottom
    '''
    length = len(sweepdata.items())

    for index, pcs_data_plot in enumerate(sweepdata.values()):
        plot_patchClampSeries(axes[index], pcs_data_plot)

        if addXTicks:
            axes[index].set_xlabel('%s [%s]' % (pcs_data_plot.axis['x'],
                                                pcs_data_plot.unit['x']))
            axes[index].xaxis.set_tick_params(labelbottom=True)

    for axis in axes[length:]:
        axis.remove()


def createPDF(nwbfile, outfile):
    '''
    convert a NeurodataWithoutBorders file to a PortableDocumentFile
    '''

    mplstyle.use(['ggplot', 'fast'])

    sweeps = gather_sweeps(nwbfile)

    with PdfPages(outfile) as pdf:
        for cycle_id in sweeps:
            sweep = sweeps.get(cycle_id)

            nacquisition = sweep.num_acquisition()
            nstimulus = sweep.num_stimulus()
            ncols = max(nacquisition, nstimulus)

            fig, axes = plt.subplots(nrows=2, ncols=ncols, sharex='row',
                                     num=cycle_id, squeeze=False)

            plot_sweepdata(sweep.get_stimulus(), axes[0][:])
            plot_sweepdata(sweep.get_acquisition(), axes[1][:], addXTicks=True)

            fig.suptitle("Sweep %s" % cycle_id)
            fig.set_size_inches(8.27, 11.69)  # a4 portrait

            plt.subplots_adjust(wspace=0.33)
            pdf.savefig(fig)
            plt.close()

        d = pdf.infodict()
        d['Title'] = nwbfile
        d['Creator'] = '/AllenInstitute/ipfx/nwb_to_pdf.py using matplotlib'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("nwbfiles", help="Path to input NWB files.", type=str,
                        nargs="+")
    args = parser.parse_args()

    for nwbfile in args.nwbfiles:
        outfile = os.path.splitext(nwbfile)[0] + ".pdf"
        print(f"Creating PDF for {nwbfile}")
        createPDF(nwbfile, outfile)


if __name__ == "__main__":
    main()
