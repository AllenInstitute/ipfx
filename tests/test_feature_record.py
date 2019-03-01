import ipfx.feature_record as fr


def test_nan_get():

    a = {}
    v = fr.nan_get(a, 'fish')
    assert v is None

    a = {'fish': 1}
    v = fr.nan_get(a, 'fish')
    assert v == 1

    a = {'fish': float("nan")}
    v = fr.nan_get(a, 'fish')
    assert v is None


def test_add_features_to_record():

    feature_names = ["foo", "bar", "karabas"]
    feature_data = {"foo": 56.0, "bar": 120.67, "baz": 0.1}
    test_record = dict()
    fr.add_features_to_record(feature_names, feature_data, test_record, postfix="suffix")
    expected_record = {"foo_suffix": 56.0, "bar_suffix": 120.67, "karabas_suffix": None}

    assert expected_record == test_record
