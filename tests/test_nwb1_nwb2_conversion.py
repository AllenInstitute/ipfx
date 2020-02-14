import os
import pytest

from ipfx.x_to_nwb.NWBConverter import NWBConverter
from .helpers_for_tests import diff_h5, validate_nwb
from ipfx.bin.run_nwb1_to_nwb2_conversion import make_nwb2_file_name
from hdmf import Container
from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
import numpy as np
import datetime
from pynwb import NWBFile, NWBHDF5IO

pytestmark = pytest.mark.skip("because pynwb changes break tests, but pynwb is planning to roll back")

class TestNWBConverter(NWBConverter):

    @staticmethod
    def undefined_object_id(self):
        """
        Monkey patching object_id property to set to a fixed value. overriding
        Reassigning Container.object_id is needed for regression testing because
        object_id is unique to created object rather than to data.

        Parameters
        ----------
        self

        Returns
        -------

        """
        return "Undefined"

    def __init__(self, input_file, output_file):

        Container.object_id = property(self.undefined_object_id)

        NWBConverter.__init__(self, input_file, output_file)

@pytest.mark.requires_inhouse_data
@pytest.mark.parametrize('NWB_file_inhouse', ['Pvalb-IRES-Cre;Ai14-406663.04.01.01.nwb',
                                              'H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_file_level_regressions(NWB_file_inhouse,tmpdir_factory):

    nwb1_file_name = NWB_file_inhouse
    base_name = os.path.basename(nwb1_file_name)

    test_dir = os.path.dirname(nwb1_file_name)
    test_nwb2_file_name = make_nwb2_file_name(test_dir,base_name)

    temp_dir = str(tmpdir_factory.mktemp("nwb_conversions"))
    temp_nwb2_file_name = make_nwb2_file_name(temp_dir,base_name)

    assert os.path.isfile(nwb1_file_name)
    assert os.path.isfile(test_nwb2_file_name)

    TestNWBConverter(input_file=nwb1_file_name,
                     output_file=temp_nwb2_file_name,
                     )

    assert validate_nwb(temp_nwb2_file_name) == []
    assert diff_h5(temp_nwb2_file_name,test_nwb2_file_name) == 0


@pytest.fixture
def nwb_filename(tmpdir_factory):
    nwb = tmpdir_factory.mktemp("test").join("test.nwb")
    return str(nwb)


def test_stimulus_round_trip(nwb_filename):

    nwbfile = NWBFile(
        session_description='test ephys',
        identifier='session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )
    device = nwbfile.create_device(name='electrode_0')

    electrode = nwbfile.create_ic_electrode(name="elec0",
                                             description=' some kind of electrode',
                                             device=device)

    data = np.array([1., 3.76, 0., 67, -2.89])
    meta_data = {"name":"test_stimulus_sweep",
                 "sweep_number": 4,
                 "unit": "A",
                 "gain": 32.0,
                 "resolution": 1.0,
                 "conversion": 1.0E-3,
                 "starting_time": 1.5,
                 "rate": 7000.0,
                 "stimulus_description": "STIMULUS_CODE"
                 }

    time_series = CurrentClampStimulusSeries(data=data,
                                             electrode=electrode,
                                             **meta_data
                                             )

    nwbfile.add_stimulus(time_series)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()

    time_series_in = nwbfile_in.get_stimulus(meta_data["name"])

    assert np.allclose(data,time_series_in.data)
    for k,v in meta_data.items():
        assert getattr(time_series_in, k) == v


def test_acquisition_round_trip(nwb_filename):

    nwbfile = NWBFile(
        session_description='test ephys',
        identifier='session_uuid',
        session_start_time=datetime.datetime.now(),
        file_create_date=datetime.datetime.now()
    )
    device = nwbfile.create_device(name='electrode_0')

    electrode = nwbfile.create_ic_electrode(name="elec0",
                                            description=' some kind of electrode',
                                            device=device)

    data = np.array([1., 3.76, 0., 67, -2.89])
    meta_data = {"name":"test_acquisition_sweep",
                 "sweep_number": 4,
                 "unit": "V",
                 "gain": 32.0,
                 "resolution": 1.0,
                 "conversion": 1.0E-3,
                 "starting_time": 1.5,
                 "rate": 7000.0,
                 "stimulus_description": "STIMULUS_CODE"
                 }

    time_series = VoltageClampStimulusSeries(data=data,
                                             electrode=electrode,
                                             **meta_data
                                             )

    nwbfile.add_acquisition(time_series)

    with NWBHDF5IO(nwb_filename, mode='w') as io:
        io.write(nwbfile)
    nwbfile_in = NWBHDF5IO(nwb_filename, mode='r').read()
    time_series_in = nwbfile_in.get_acquisition(meta_data["name"])

    assert np.allclose(data,time_series_in.data)
    for k,v in meta_data.items():
        assert getattr(time_series_in, k) == v

