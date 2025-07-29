"""Tests for NWB2Sink. 

See test_cli for integration tests. There are some nwbfile-on-disk handling 
components which are tested there rather than here, namely:
    _initial_load_nwbfile
    _reload_nwbfile
    _commit_nwb_changes
"""
from datetime import datetime, timezone, timedelta
import io
import os

import pytest
import numpy as np
import pynwb
import h5py

from ipfx.attach_metadata.sink import nwb2_sink


@pytest.fixture
def nwbfile():
    _nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    dev = pynwb.device.Device(name="my_device")
    _nwbfile.add_device(dev)
    ice = pynwb.icephys.IntracellularElectrode(
        name="my_electrode",
        device=dev,
        description=""
    )
    _nwbfile.add_icephys_electrode(ice)
    series = pynwb.icephys.CurrentClampSeries(
          name="a current clamp", 
          data=[1, 2, 3], 
          starting_time=0.0, 
          rate=1.0,
          gain=1.0,
          electrode=ice,
          sweep_number=12
    )
    _nwbfile.add_acquisition(series, use_sweep_table=True)
    _nwbfile.subject = pynwb.file.Subject()

    return _nwbfile


def test_get_single_ic_electrode(nwbfile):
    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile

    obt = sink._get_single_ic_electrode()
    assert obt.name == "my_electrode"


def test_get_sweep_series(nwbfile):
    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile

    obt = sink._get_sweep_series(12)[0]
    assert np.allclose(obt.data[:], [1, 2, 3])


@pytest.mark.parametrize("set_none", [True, False])
def test_get_subject(nwbfile, set_none):
    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile
    if set_none:
        nwbfile.subject = None

    subject = sink._get_subject()
    subject.subject_id = "foo"
    assert nwbfile.subject.subject_id == "foo"


def test_serialize(tmpdir_factory, nwbfile):
    out_path = os.path.join(
        str(tmpdir_factory.mktemp("test_serialize")),
        "out.nwb"
    )

    sink = nwb2_sink.Nwb2Sink(None)
    sink._data = io.BytesIO()
    sink._h5_file = h5py.File(sink._data, "w")
    sink._nwb_io = pynwb.NWBHDF5IO(
        path=sink._h5_file.filename,
        mode="w",
        file=sink._h5_file
    )
    sink.nwbfile = nwbfile

    sink.serialize({"output_path": out_path})

    with pynwb.NWBHDF5IO(out_path, "r", load_namespaces=True) as reader:
        obt = reader.read()
        assert obt.identifier == "test session"


def test_roundtrip(tmpdir_factory, nwbfile):
    tmpdir = str(tmpdir_factory.mktemp("test_serialize"))
    first_path = os.path.join(tmpdir, "first.nwb")
    second_path = os.path.join(tmpdir, "second.nwb")

    with pynwb.NWBHDF5IO(first_path, "w") as writer:
        writer.write(nwbfile)

    sink = nwb2_sink.Nwb2Sink(first_path)
    sink.register("institution", "AIBS")
    sink.serialize({"output_path": second_path})

    with pynwb.NWBHDF5IO(second_path, "r", load_namespaces=True) as reader:
        obt = reader.read()
        assert obt.institution == "AIBS"

@pytest.mark.parametrize("name, value, sweep_id, getter, expected", [
    ["subject_id", "mouse01", None, lambda f: f.subject.subject_id, "mouse01"],
    ["institution", "aibs", None, lambda f: f.institution, "aibs"],
    ["session_id", "12345", None, lambda f: f.session_id, "12345"],
    ["age", "P56D", None, lambda f: f.subject.age, "P56D"],
    ["date_of_birth", "2019-05-12 17:00:00 -0700", None, 
        lambda f: f.subject.date_of_birth, datetime(
            2019, 5, 12, 17, 0, tzinfo=timezone(timedelta(hours=-7))
        )
    ],
    ["genotype", "Gad2-IRES-Cre/wt;Ai14(RCL-tdT)/wt", None, 
        lambda f: f.subject.genotype, "Gad2-IRES-Cre/wt;Ai14(RCL-tdT)/wt"
    ],
    ["sex", "F", None, lambda f: f.subject.sex, "F"],
    ["species", "Mus musculus", None, lambda f: f.subject.species, 
        "Mus musculus"
    ],
])
def test_register(
        tmpdir_factory, nwbfile, name, value, sweep_id, getter, expected
):
    tmpdir = tmpdir_factory.mktemp("test_register")
    path = os.path.join(str(tmpdir), "test.nwb")

    sink = nwb2_sink.Nwb2Sink(None)
    sink.nwbfile = nwbfile

    sink.register(name, value, sweep_id)

    with pynwb.NWBHDF5IO(path, "w") as writer:
        writer.write(nwbfile)

    with pynwb.NWBHDF5IO(path, "r") as reader:
        obt_nwb = reader.read()

    obtained = getter(obt_nwb)
    assert obtained == expected
