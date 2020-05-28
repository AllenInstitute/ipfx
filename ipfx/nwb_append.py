from pathlib import Path
from typing import (
    List, Dict, Optional, Union
)

import shutil

import pynwb
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import TimeSeries
from pynwb import ProcessingModule
import numpy as np


PathLike = Union[
    str,
    Path
]


def append_spike_times(input_nwb_path: PathLike,
                       sweep_spike_times: Dict[int, List[float]],
                       output_nwb_path: Optional[PathLike] = None):
    """
        Appends spiketimes to an nwb2 file

        Paramters
        ---------

        input_nwb_path: location of input nwb file without spiketimes

        spike_times: Dict of sweep_num: spiketimes

        output_nwb_path: optional location to write new nwb file with
                         spiketimes, otherwise appends spiketimes to
                         input file

    """

    # Copy to new location
    if output_nwb_path and output_nwb_path != input_nwb_path:
        shutil.copy(input_nwb_path, output_nwb_path)
        nwb_path = output_nwb_path
    else:
        nwb_path = input_nwb_path

    nwb_io = pynwb.NWBHDF5IO(nwb_path, mode='a', load_namespaces=True)
    nwbfile = nwb_io.read()

    # Create spike module
    spike_module = ProcessingModule(name='spikes',
                                    description='detected spikes')
    for sweep_num, spike_times in sweep_spike_times.items():
        wrapped_spike_times = H5DataIO(data=np.asarray(spike_times),
                                       compression=True)
        ts = TimeSeries(timestamps=wrapped_spike_times,
                        unit='seconds',
                        data=wrapped_spike_times,
                        name=f"Sweep_{sweep_num}")
        spike_module.add_data_interface(ts)

    nwbfile.add_processing_module(spike_module)

    nwb_io.write(nwbfile)
    nwb_io.close()
