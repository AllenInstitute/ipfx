import os
import numbers
import urllib2
import shutil

import pytest
from pytest import approx

import h5py
import numpy as np

from ipfx.nwb_reader import create_nwb_reader, NwbMiesReader, NwbPipelineReader, NwbXReader
from allensdk.api.queries.cell_types_api import CellTypesApi


@pytest.fixture()
def fetch_pipeline_file():
    specimen_id = 595570553
    nwb_file = '{}.nwb'.format(specimen_id)
    if not os.path.exists(nwb_file):
        ct = CellTypesApi()
        ct.save_ephys_data(specimen_id, nwb_file)


@pytest.fixture()
def fetch_DAT_NWB_file():
    output_filepath = 'H18.28.015.11.14.nwb'
    if not os.path.exists(output_filepath):

        BASE_URL = "https://www.byte-physics.de/Downloads/allensdk-test-data/"

        response = urllib2.urlopen(BASE_URL + output_filepath)
        with open(output_filepath, "wb") as out_file:
            shutil.copyfileobj(response, out_file)


def compare_dicts(d_ref, d):
    # pytest does not support passing in dicts of numpy arrays with strings
    # See https://github.com/pytest-dev/pytest/issues/4079 and
    # https://github.com/pytest-dev/pytest/issues/4079
    assert sorted(d_ref.keys()) == sorted(d.keys())
    for k, v in d_ref.items():
        if isinstance(v, np.ndarray):
            array_ref = d_ref[k]
            array = d[k]

            assert len(array) == len(array_ref)
            for index in range(len(array)):
                if isinstance(array[index], (str, unicode)):
                    assert array[index] == array_ref[index]
                else:
                    assert array[index] == approx(array_ref[index])
        else:
            value_ref = d_ref[k]
            value = d[k]

            if isinstance(value_ref, numbers.Number):
                assert value_ref == approx(value, nan_ok=True)
            else:
                assert value_ref == value


def test_raises_on_missing_file():
    with pytest.raises(IOError):
        create_nwb_reader('I_DONT_EXIST.nwb')


def test_raises_on_empty_h5_file():
    filename = 'empty.nwb'

    with h5py.File(filename, 'w'):
        pass

    with pytest.raises(ValueError, match=r'unknown NWB major'):
        create_nwb_reader(filename)


def test_valid_v1_but_unknown_sweep_naming():
    filename = 'invalid_sweep_naming_convention.nwb'

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

    with pytest.raises(ValueError, match=r'sweep naming convention'):
        create_nwb_reader(filename)


def test_valid_v1_skeleton_MIES():
    filename = 'valid_v1_MIES.nwb'

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

        dset = fh.create_dataset(
            "acquisition/timeseries/data_00000", (1, ), dtype="f")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbMiesReader)


def test_valid_v1_skeleton_Pipeline():
    filename = 'valid_v1_Pipeline.nwb'

    with h5py.File(filename, 'w') as fh:
        dset = fh.create_dataset("nwb_version", (1,), dtype="S5")
        dset[:] = str("NWB-1")

        dset = fh.create_dataset(
            "acquisition/timeseries/Sweep_0", (1, ), dtype="f")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbPipelineReader)


def test_valid_v1_skeleton_X_NWB():
    filename = 'valid_v2.nwb'

    with h5py.File(filename, 'w') as fh:
        fh.attrs["nwb_version"] = str("2")

    reader = create_nwb_reader(filename)
    assert isinstance(reader, NwbXReader)


def test_valid_v1_full_Pipeline(fetch_pipeline_file):
    reader = create_nwb_reader('595570553.nwb')
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

    assert reader.get_stim_code("Sweep_10") == "Short Square"

    sweep_attrs_ref = {
        u'ancestry': np.array(['TimeSeries', 'PatchClampSeries', 'CurrentClampSeries'], dtype='|S18'),
        u'comments': u'',
        u'description': u'',
        u'help': u'Voltage recorded from cell during current-clamp recording',
        u'missing_fields': np.array(['gain'], dtype='|S4'),
        u'neurodata_type': u'TimeSeries',
        u'source': u''}

    sweep_attrs = reader.get_sweep_attrs("Sweep_10")
    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (37500, 101149),
                      'response': None,
                      'sampling_rate': 50000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Amps'}

    sweep_data = reader.get_sweep_data(10)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


def test_valid_v1_full_MIES_1():
    reader = create_nwb_reader(os.path.join(os.path.dirname(__file__), 'data',
                               'UntitledExperiment-2018_12_03_234957-compressed.nwb'))
    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code("data_00000_AD0") == "StimulusSetA"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment':  None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs("data_00000_AD0")
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (0, 188000),
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


def test_valid_v1_full_MIES_2():
    reader = create_nwb_reader(os.path.join(os.path.dirname(__file__), 'data',
                               'H18.03.315.11.11.01.05.nwb'))
    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code("data_00000_AD0") == "EXTPSMOKET180424"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment': None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction', u'whole_cell_capacitance_comp',
                                                    u'whole_cell_series_resistance_comp'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs("data_00000_AD0")
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (0, 65999),
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


def test_valid_v1_full_MIES_3():
    reader = create_nwb_reader(os.path.join(os.path.dirname(__file__), 'data',
                               'Sst-IRES-CreAi14-395722.01.01.01.nwb'))

    assert isinstance(reader, NwbMiesReader)

    sweep_names_ref = [u'data_00000_AD0']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("data_00000_AD0") == 0

    assert reader.get_stim_code("data_00000_AD0") == "EXTPSMOKET180424"

    # ignore very long comment
    sweep_attrs_ref = {u'ancestry': np.array([u'TimeSeries', u'PatchClampSeries', u'VoltageClampSeries'], dtype=object),
                       u'comment': None,
                       u'description': u'PLACEHOLDER',
                       u'missing_fields': np.array([u'resistance_comp_bandwidth', u'resistance_comp_correction',
                                                    u'resistance_comp_prediction', u'whole_cell_capacitance_comp',
                                                    u'whole_cell_series_resistance_comp'], dtype=object),
                       u'neurodata_type': u'TimeSeries',
                       u'source': u'Device=ITC18USB_Dev_0;Sweep=0;AD=0;ElectrodeNumber=0;ElectrodeName=0'}

    sweep_attrs = reader.get_sweep_attrs("data_00000_AD0")
    sweep_attrs['comment'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (0, 65999),
                      'response': None,
                      'sampling_rate': 200000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


def test_valid_v2_full_ABF():
    reader = create_nwb_reader(os.path.join(os.path.dirname(__file__), 'data',
                               '2018_03_20_0005.nwb'))
    assert isinstance(reader, NwbXReader)

    sweep_names_ref = [u'index_0', u'index_1']

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("index_0") == 0

    assert reader.get_stim_code("index_0") == "RAMP1"

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

    sweep_attrs = reader.get_sweep_attrs("index_0")
    sweep_attrs['description'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (0, 899999),
                      'response': None,
                      'sampling_rate': 50000.0,
                      'stimulus': None,
                      'stimulus_unit': 'Amps'}

    sweep_data = reader.get_sweep_data(0)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data


def test_valid_v2_full_DAT(fetch_DAT_NWB_file):

    reader = create_nwb_reader('H18.28.015.11.14.nwb')
    assert isinstance(reader, NwbXReader)

    sweep_names_ref = ['index_{:02d}'.format(x) for x in range(0, 78)]

    sweep_names = reader.get_sweep_names()
    assert sorted(sweep_names_ref) == sorted(sweep_names)

    assert reader.get_pipeline_version() == (0, 0)

    assert reader.get_sweep_number("index_00") == 10101

    assert reader.get_stim_code("index_00") == "extpinbath"

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

    sweep_attrs = reader.get_sweep_attrs("index_00")
    sweep_attrs['description'] = None

    compare_dicts(sweep_attrs_ref, sweep_attrs)

    # assume the data itself is correct and replace it with None
    sweep_data_ref = {'index_range': (0, 199999),
                      'response': None,
                      'sampling_rate': 200000.00000000003,
                      'stimulus': None,
                      'stimulus_unit': 'Volts'}

    sweep_data = reader.get_sweep_data(10101)
    sweep_data['response'] = None
    sweep_data['stimulus'] = None

    assert sweep_data_ref == sweep_data
