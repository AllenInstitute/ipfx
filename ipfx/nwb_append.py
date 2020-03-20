import os
import pynwb
from allensdk.core.nwb_data_set import NwbDataSet
from ipfx.dataset.create import get_nwb_version
from pynwb import TimeSeries
from pynwb import ProcessingModule


class NwbAppender(object):

    def __init__(self, nwb_file_name):
        if os.path.isfile(nwb_file_name):
            self.nwb_file_name = nwb_file_name
        else:
            raise FileNotFoundError(f"Cannot locate {nwb_file_name}")

    def add_spike_times(self, sweep_num, spike_times):
        raise NotImplementedError

    def set_spike_time(self, sweep_spike_times):
        raise NotImplementedError


class Nwb1Appender(NwbAppender):

    def __init__(self, nwb_file_name):
        NwbAppender.__init__(self, nwb_file_name)
        self.nwbfile = NwbDataSet(self.nwb_file_name)

    def add_spike_times(self, sweep_spike_times):

        for sweep_num, spike_times in sweep_spike_times.items():
            self.nwbfile.set_spike_times(sweep_num, spike_times)


class Nwb2Appender(NwbAppender):

    def __init__(self, nwb_file_name):
        NwbAppender.__init__(self, nwb_file_name)

        io = pynwb.NWBHDF5IO(self.nwb_file_name, 'a')
        self.nwbfile = io.read()
        io.close()

    def add_spike_times(self, sweep_spike_times):

        spike_module = ProcessingModule(name='spikes',
                                        description='detected spikes')

        for sweep_num, spike_times in sweep_spike_times.items():
            ts = TimeSeries(timestamps=spike_times, name=f"Sweep_{sweep_num}")
            spike_module.add_data_interface(ts)

        self.nwbfile.add_processing_module(spike_module)

        io = pynwb.NWBHDF5IO(self.nwb_file_name, 'w')
        io.write(self.nwbfile)
        io.close()


def create_nwb_appender(nwb_file):
    """Create an appropriate writer of the nwb_file

    Parameters
    ----------
    nwb_file: str file name

    Returns
    -------
    writer object
    """

    if os.path.isfile(nwb_file):
        nwb_version = get_nwb_version(nwb_file)
    else:
        raise FileNotFoundError(f"Cannot locate {nwb_file}")

    if nwb_version["major"] == 2:
        return Nwb2Appender(nwb_file)
    elif nwb_version["major"] == 1 or nwb_version["major"] == 0:
        return Nwb1Appender(nwb_file)
    else:
        raise ValueError(
            "Unsupported or unknown NWB major version {} ({})".format(
                nwb_version["major"], nwb_version["full"])
        )
