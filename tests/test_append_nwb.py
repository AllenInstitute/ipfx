import pytest
import datetime
import pynwb
from unittest.mock import patch

import numpy as np

from ipfx.bin.run_feature_extraction import embed_spike_times
from ipfx.dataset.ephys_nwb_data import EphysNWBData


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


with patch.multiple(
    EphysNWBData,
    __abstractmethods__=[]
):

    class JustConcrete(EphysNWBData):
        pass


@pytest.mark.parametrize('make_skeleton_nwb_file', [make_skeleton_nwb2_file])
def test_embed_spike_times_into_nwb(make_skeleton_nwb_file, tmpdir_factory):

    sweep_spike_times = {
        3: [56.0, 44.6, 661.1],
        4: [156.0, 144.6, 61.1, 334.944]
    }

    tmp_dir = tmpdir_factory.mktemp("embed_spikes_into_nwb")
    input_nwb_file_name = str(tmp_dir.join("input.nwb"))
    output_nwb_file_name = str(tmp_dir.join("output.nwb"))

    make_skeleton_nwb_file(input_nwb_file_name)

    embed_spike_times(
        input_nwb_file_name, output_nwb_file_name, sweep_spike_times
    )

    reader = JustConcrete(output_nwb_file_name, None)
    for sweep_num, spike_times in sweep_spike_times.items():
        assert np.allclose(reader.get_spike_times(sweep_num), spike_times)
