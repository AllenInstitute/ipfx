import pytest
import datetime
import pynwb
import numpy as np

from ipfx.nwb_append import append_spike_times


def make_skeleton_nwb2_file(nwb2_file_name):

    nwbfile = pynwb.NWBFile(
        session_description='test icephys',
        identifier='session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )

    device = nwbfile.create_device(name='electrode_0')
    nwbfile.create_ic_electrode(
        name="elec0",
        description='intracellular electrode',
        device=device
    )

    io = pynwb.NWBHDF5IO(nwb2_file_name, 'w')
    io.write(nwbfile)
    io.close()


def test_embed_spike_times_into_nwb(tmpdir_factory):

    sweep_spike_times = {
        3: [56.0, 44.6, 661.1],
        4: [156.0, 144.6, 61.1, 334.944]
    }

    tmp_dir = tmpdir_factory.mktemp("embed_spikes_into_nwb")
    input_nwb_file_name = str(tmp_dir.join("input.nwb"))
    output_nwb_file_name = str(tmp_dir.join("output.nwb"))

    make_skeleton_nwb2_file(input_nwb_file_name)

    append_spike_times(input_nwb_file_name,
                       sweep_spike_times,
                       output_nwb_path=output_nwb_file_name)

    with pynwb.NWBHDF5IO(output_nwb_file_name, mode='r', load_namespaces=True) as nwb_io:
        nwbfile = nwb_io.read()

        spikes = nwbfile.get_processing_module('spikes')
        for sweep_num, spike_times in sweep_spike_times.items():
            sweep_spikes = spikes.get_data_interface(f"Sweep_{sweep_num}").timestamps
            assert np.allclose(sweep_spikes, spike_times)
