import pytest
import datetime
import pynwb
import h5py

import numpy as np

from ipfx.bin.run_feature_extraction import embed_spike_times


def make_skeleton_nwb1_file(nwb1_file_name):

    with h5py.File(nwb1_file_name, 'w') as fh:
        dt = h5py.special_dtype(vlen=bytes)
        dset = fh.create_dataset("nwb_version", (1,), dtype=dt)
        dset[:] = "NWB-1"
        fh.create_group("acquisition/timeseries")
        fh.create_group("analysis")


def make_skeleton_nwb2_file(nwb2_file_name):

    nwbfile = pynwb.NWBFile(
        session_description='test icephys',
        identifier='session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    device = nwbfile.create_device(name='electrode_0')
    electrode = nwbfile.create_ic_electrode(name="elec0",
                                            description='intracellular electrode',
                                            device=device)

    io = pynwb.NWBHDF5IO(nwb2_file_name, 'w')
    io.write(nwbfile)
    io.close()


@pytest.mark.parametrize('make_skeleton_nwb_file', (make_skeleton_nwb1_file, make_skeleton_nwb2_file))
def test_embed_spike_times_into_nwb(make_skeleton_nwb_file, tmpdir_factory):

    sweep_spike_times = {
        3: [56.0, 44.6, 661.1],
        4: [156.0, 144.6, 61.1, 334.944]
    }

    tmp_dir = tmpdir_factory.mktemp("embed_spikes_into_nwb")
    input_nwb_file_name = str(tmp_dir.join("input.nwb"))
    output_nwb_file_name = str(tmp_dir.join("output.nwb"))

    make_skeleton_nwb_file(input_nwb_file_name)

    embed_spike_times(input_nwb_file_name, output_nwb_file_name, sweep_spike_times)

    # TODO replace with nwb access
    nwb_data = nwb_reader.create_nwb_reader(output_nwb_file_name)

    for sweep_num, spike_times in sweep_spike_times.items():
        assert np.allclose(nwb_data.get_spike_times(sweep_num), spike_times)


