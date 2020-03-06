"""
"""
from datetime import datetime

import numpy as np
import pynwb

from ipfx.attach_metadata.sink import nwb2_sink


def test_set_container_sources():

    ts = pynwb.TimeSeries(
          name="a timeseries", 
          data=[1, 2, 3], 
          starting_time=0.0, 
          rate=1.0
        )
    subj = pynwb.file.Subject()
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    nwbfile.add_acquisition(ts)
    nwbfile.subject = subj

    nwb2_sink.set_container_sources(nwbfile, "foo")
    assert nwbfile.container_source == "foo"
    assert ts.container_source == "foo"
    assert subj.container_source == "foo"


def test_get_single_ic_electrode():
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    dev = pynwb.device.Device(name="my_device")
    nwbfile.add_device(dev)
    nwbfile.add_ic_electrode(pynwb.icephys.IntracellularElectrode(
        name="my_electrode",
        device=dev,
        description=""
    ))

    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile

    obt = sink._get_single_ic_electrode()
    assert obt.name == "my_electrode"


def test_get_sweep_series():
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    dev = pynwb.device.Device(name="my_device")
    nwbfile.add_device(dev)
    ice = pynwb.icephys.IntracellularElectrode(
        name="my_electrode",
        device=dev,
        description=""
    )
    nwbfile.add_ic_electrode(ice)
    series = pynwb.icephys.CurrentClampSeries(
          name="a timeseries", 
          data=[1, 2, 3], 
          starting_time=0.0, 
          rate=1.0,
          gain=1.0,
          electrode=ice,
          sweep_number=12
    )
    nwbfile.add_acquisition(series)
    # nwbfile.sweep_table.add_entry(series)

    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile

    obt = sink._get_sweep_series(12)[0]
    assert np.allclose(obt.data[:], [1, 2, 3])