import pytest
import ipfx.data_set_features as dsft
from ipfx.error import FeatureError


def test_select_subthreshold_min_amplitude():

    a = [ 10, 10, 10, 10 ]
    with pytest.raises(IndexError):
        min_amp, delta = dsft.select_subthreshold_min_amplitude(a)

    a = [10, 20, 30, 40]
    with pytest.raises(FeatureError):
        min_amp, delta = dsft.select_subthreshold_min_amplitude(a)

    a = [10, 30, 50, 70]
    min_amp, delta = dsft.select_subthreshold_min_amplitude(a)
    assert delta == 20
    assert min_amp == -100

    a = [10, 50, 90, 120]
    with pytest.raises(FeatureError):
        min_amp, delta = dsft.select_subthreshold_min_amplitude(a)

    a = [10, 50, 90, 130]
    min_amp, delta = dsft.select_subthreshold_min_amplitude(a)
    assert delta == 40
    assert min_amp == -200

    a = [10, 50, 50, 90, 90, 130]
    min_amp, delta = dsft.select_subthreshold_min_amplitude(a)
    assert delta == 40
    assert min_amp == -200

    a = [10, 30, 30, 50, 90]
    min_amp, delta = dsft.select_subthreshold_min_amplitude(a)
    assert delta == 20
    assert min_amp == -100

    a = [10, 50, 50, 70, 90]
    min_amp, delta = dsft.select_subthreshold_min_amplitude(a)
    assert delta == 20
    assert min_amp == -100




def test_nan_get():

    a = {}
    v = dsft.nan_get(a, 'fish')
    assert v == None

    a = { 'fish': 1 }
    v = dsft.nan_get(a, 'fish')
    assert v == 1

    a = { 'fish': float("nan") }
    v = dsft.nan_get(a, 'fish')
    assert v == None

