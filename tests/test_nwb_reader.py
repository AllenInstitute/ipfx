import os
import pytest
import h5py
import numpy as np
from ipfx.nwb_reader import create_nwb_reader, NwbMiesReader, NwbPipelineReader, NwbXReader
from helpers_for_tests import compare_dicts
from allensdk.api.queries.cell_types_api import CellTypesApi

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture(scope="session", params=[595570553])
def fetch_pipeline_file(request):
    specimen_id = request.param
    nwb_file_name = '{}.nwb'.format(specimen_id)

    nwb_file_full_path = os.path.join(TEST_DATA_PATH, nwb_file_name)

    if not os.path.exists(nwb_file_full_path):
        ct = CellTypesApi()
        ct.save_ephys_data(specimen_id, nwb_file_full_path)

    return nwb_file_full_path


def test_raises_on_missing_file():
    with pytest.raises(IOError):
        create_nwb_reader('I_DONT_EXIST.nwb')


def test_raises_on_empty_h5_file():

    filename = os.path.join(TEST_DATA_PATH, "empty.nwb")

    with h5py.File(filename, 'w'):
        pass

    with pytest.raises(ValueError, match=r'unknown NWB major'):
        create_nwb_reader(filename)


def test_valid_v1_but_unknown_sweep_naming():

    filename = os.path.join(TEST_DATA_PATH, 'invalid_sweep_naming_convention.nwb')

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

    with pytest.raises(ValueError, match=r'sweep naming convention'):
        create_nwb_reader(filename)


def test_valid_v1_with_no_sweeps():

    filename = os.path.join(TEST_DATA_PATH, 'no_sweeps.nwb')

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")
        fh.create_group("acquisition/timeseries")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbMiesReader)

@pytest.mark.parametrize('NWB_file', ['2018_03_20_0005.nwb'], indirect=True)
def test_get_recording_date(NWB_file):
    reader = create_nwb_reader(NWB_file)

    assert "2018-03-20 20:59:48" == reader.get_recording_date()


def test_valid_v1_skeleton_MIES():
    filename = os.path.join(TEST_DATA_PATH, 'valid_v1_MIES.nwb')

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

        dset = fh.create_dataset(
            "acquisition/timeseries/data_00000", (1, ), dtype="f")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbMiesReader)


def test_valid_v1_skeleton_Pipeline():
    filename = os.path.join(TEST_DATA_PATH, 'valid_v1_Pipeline.nwb')

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

        dset = fh.create_dataset(
            "acquisition/timeseries/Sweep_0", (1, ), dtype="f")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbPipelineReader)


def test_valid_v1_skeleton_X_NWB():
    filename = os.path.join(TEST_DATA_PATH, 'valid_v2.nwb')

    with h5py.File(filename, 'w') as fh:
        fh.attrs["nwb_version"] = str("2")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbXReader)


@pytest.mark.parametrize('NWB_file', ["500844779.nwb", "509604657.nwb"], indirect=True)
def test_assumed_sweep_number_fallback(NWB_file):

    reader = create_nwb_reader(NWB_file)
    assert isinstance(reader, NwbPipelineReader)

    assert reader.get_sweep_number("Sweep_10") == 10


def test_valid_v1_full_Pipeline(fetch_pipeline_file):
    reader = create_nwb_reader(fetch_pipeline_file)
    assert isinstance(reader, NwbPipelineReader)

    sweep_names_ref = [u'Sweep_10',
                       u'Sweep_12',
                       u'Sweep_13',
                       u'Sweep_14',
                       u'Sweep_15',
                       u'Sweep_16',
                       u'Sweep_17',
                       u'Sweep_19',
                       u'Sweep_20',
                       u'Sweep_25',
                       u'Sweep_28',
                       u'Sweep_29',
                       u'Sweep_30',
                       u'Sweep_32',
                       u'Sweep_33',
                       u'Sweep_34',
                       u'Sweep_35',
                       u'Sweep_36',
                       u'Sweep_37',
                       u'Sweep_38',
                       u'Sweep_39',
                       u'Sweep_40',
                       u'Sweep_41',
                       u'Sweep_42',
                       u'Sweep_43',
                       u'Sweep_44',
                       u'Sweep_45',
                       u'Sweep_46',
                       u'Sweep_47',
                       u'Sweep_5',
                       u'Sweep_51',
                       u'Sweep_52',
                       u'Sweep_53',
                       u'Sweep_54',
                       u'Sweep_55',
                       u'Sweep_57',
                       u'Sweep_58',
                       u'Sweep_59',
                       u'Sweep_6',
                       u'Sweep_61',
                       u'Sweep_62',
                       u'Sweep_63',
                       u'Sweep_64',
                       u'Sweep_65',
                       u'Sweep_66',
                       u'Sweep_67',
                       u'Sweep_68',
                       u'Sweep_69',
                       u'Sweep_7',
                       u'Sweep_70',
                       u'Sweep_74',
                       u'Sweep_8',
                       u'Sweep_9']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (1, 0)

    assert reader.get_sweep_number("Sweep_10") == 10

    assert reader.get_stim_code(10) == "Short Square"

    sweep_attrs_ref = {
        u'ancestry': np.array(['TimeSeries', 'PatchClampSeries', 'CurrentClampSeries'], dtype='|S18'),
        u'comments': u'',
        u'description': u'',
        u'help': u'Voltage recorded from cell during current-clamp recording',
        u'missing_fields': np.array(['gain'], dtype='|S4'),
        u'neurodata_type': u'TimeSeries',
        u'source': u''}

    sweep_attrs = reader.get_sweep_attrs(10)
    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 50000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Amps'}

    sweep_data = reader.get_sweep_data(10)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


@pytest.mark.parametrize('NWB_file', ['UntitledExperiment-2018_12_03_234957-compressed.nwb'], indirect=True)
def test_valid_v1_full_MIES_1(NWB_file):

    reader = create_nwb_reader(NWB_file)

    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code(0) == "StimulusSetA"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment':  None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs(0)
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


@pytest.mark.parametrize('NWB_file', ['Pvalb-IRES-Cre;Ai14-415796.02.01.01.nwb'], indirect=True)
def test_sweep_map_sweep_numbers(NWB_file):

    sweep_numbers_ref = np.arange(0,71)
    reader = create_nwb_reader(NWB_file)

    sweep_map_table = reader.sweep_map_table
    sweep_numbers = sweep_map_table["sweep_number"].values
    print sweep_numbers_ref

    assert (sweep_numbers == sweep_numbers_ref).all()


@pytest.mark.parametrize('NWB_file', ['Pvalb-IRES-Cre;Ai14-415796.02.01.01.nwb'], indirect=True)
def test_sweep_map_sweep_0(NWB_file):

    reader = create_nwb_reader(NWB_file)
    sweep_map_ref = {'acquisition_group': u'data_00046_AD0',
                     'stimulus_group': u'data_00046_DA0',
                     'sweep_number': 0,
                     'starting_time': 2740.1590003967285}

    sweep_map = reader.get_sweep_map(0)
    assert sweep_map == sweep_map_ref


@pytest.mark.parametrize('NWB_file', ['H18.03.315.11.11.01.05.nwb'], indirect=True)
def test_valid_v1_full_MIES_2(NWB_file):

    reader = create_nwb_reader(NWB_file)
    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code(0) == "EXTPSMOKET180424"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment': None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction', u'whole_cell_capacitance_comp',
                                                    u'whole_cell_series_resistance_comp'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs(0)
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


@pytest.mark.parametrize('NWB_file', ['Sst-IRES-CreAi14-395722.01.01.01.nwb'], indirect=True)
def test_valid_v1_full_MIES_3(NWB_file):

    reader = create_nwb_reader(NWB_file)

    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code(0) == "EXTPSMOKET180424"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment': None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction', u'whole_cell_capacitance_comp',
                                                    u'whole_cell_series_resistance_comp'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs(0)
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


@pytest.mark.parametrize('NWB_file', ['2018_03_20_0005.nwb'], indirect=True)
def test_valid_v2_full_ABF(NWB_file):

    reader = create_nwb_reader(NWB_file)
    assert isinstance(reader, NwbXReader)

    sweep_names_ref = [u'index_0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("index_0") == 0

    assert reader.get_stim_code(0) == "RAMP1"

    # ignore very long description
    sweep_attrs_ref = {u'bias_current': np.nan,
                       u'bridge_balance': np.nan,
                       u'capacitance_compensation': np.nan,
                       u'comments': u'no comments',
                       u'description': None,
                       u'gain': 1.0,
                       u'help': u'Voltage recorded from cell during current-clamp recording',
                       u'namespace': u'core',
                       u'neurodata_type': u'CurrentClampSeries',
                       u'starting_time': 0.0,
                       u'stimulus_description': u'RAMP1',
                       u'sweep_number': 0}

    sweep_attrs = reader.get_sweep_attrs(0)
    sweep_attrs['description'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 50000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Amps'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


@pytest.mark.parametrize('NWB_file', ['H18.28.015.11.14.nwb'], indirect=True)
def test_valid_v2_full_DAT(NWB_file):
    reader = create_nwb_reader(NWB_file)
    assert isinstance(reader, NwbXReader)

    sweep_names_ref = ['index_{:02d}'.format(x) for x in range(0, 78)]

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("index_00") == 10101

    assert reader.get_stim_code(10101) == "extpinbath"

    # ignore very long description
    sweep_attrs_ref = {u'capacitance_fast': 0.0,
                       u'capacitance_slow': np.nan,
                       u'comments': u'no comments',
                       u'description': None,
                       u'gain': 5000000.0,
                       u'help': u'Current recorded from cell during voltage-clamp recording',
                       u'namespace': u'core',
                       u'neurodata_type': u'VoltageClampSeries',
                       u'resistance_comp_bandwidth': np.nan,
                       u'resistance_comp_correction': np.nan,
                       u'resistance_comp_prediction': np.nan,
                       u'starting_time': 3768.2174599999998,
                       u'stimulus_description': u'extpinbath',
                       u'sweep_number': 10101,
                       u'whole_cell_capacitance_comp': np.nan,
                       u'whole_cell_series_resistance_comp': np.nan}

    sweep_attrs = reader.get_sweep_attrs(10101)
    sweep_attrs['description'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {
                      'response': None,
                      'sampling_rate': 200000.00000000003,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(10101)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data
