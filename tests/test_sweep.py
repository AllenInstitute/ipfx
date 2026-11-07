from ipfx.sweep import Sweep
import pytest
import numpy as np


@pytest.fixture()
def sweep():

    i = [0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    v = [
        0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1,
        np.nan, np.nan, np.nan, np.nan, np.nan,
    ]
    sampling_rate = 2
    dt = 1. / sampling_rate
    t = np.arange(0, len(v)) * dt

    return Sweep(
        t, v, i, sampling_rate=sampling_rate, clamp_mode="CurrentClamp"
    )


def _make_nwb_epoch(
    short_name, start_idx, end_idx, treelevel=1, extra_tags=None
):
    tags = {"Type": "Epoch", "ShortName": short_name}
    if extra_tags:
        tags.update(extra_tags)
    return {
        "start_time": start_idx / 2.0,
        "stop_time": end_idx / 2.0,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "treelevel": treelevel,
        "tags": tags,
        # Mirrors what MIESNWBData.get_nwb_epochs() computes at load time --
        # Sweep itself no longer applies the prefix.
        "name": f"{Sweep.NWB_EPOCH_PREFIX}{short_name}",
    }


@pytest.fixture()
def sweep_with_nwb_epochs():

    i = [0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    v = [
        0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1,
        np.nan, np.nan, np.nan, np.nan, np.nan,
    ]
    sampling_rate = 2
    dt = 1. / sampling_rate
    t = np.arange(0, len(v)) * dt

    nwb_epochs = [
        _make_nwb_epoch("E0", 0, 3),
        _make_nwb_epoch("E1", 4, 6),
    ]

    return Sweep(
        t, v, i, sampling_rate=sampling_rate, clamp_mode="CurrentClamp",
        nwb_epochs=nwb_epochs,
    )


def test_select_epoch(sweep):

    sweep.select_epoch("sweep")
    i_sweep = sweep.i
    v_sweep = sweep.v

    sweep.select_epoch("recording")
    assert np.all(sweep.i == [0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2])
    assert np.all(sweep.v == [0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 1])

    sweep.select_epoch("sweep")
    assert np.all(sweep.i == i_sweep)
    assert np.all(sweep.v == v_sweep)


def test_set_time_zero_to_index(sweep):

    t0_idx = 7
    sweep.set_time_zero_to_index(t0_idx)

    assert np.isclose(sweep.t[t0_idx], 0.0)


def test_no_nwb_epochs_is_backward_compatible(sweep):

    assert sweep.nwb_epochs == []
    sweep.select_epoch("recording")
    assert np.all(sweep.i == [0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2])


def test_select_nwb_epoch_by_name_slices_t_v_i(sweep_with_nwb_epochs):

    sweep_with_nwb_epochs.select_epoch("nwb:E1")
    assert np.all(sweep_with_nwb_epochs.i == [0, 0, 0])
    assert np.all(sweep_with_nwb_epochs.v == [1, 0, 0])
    assert len(sweep_with_nwb_epochs.t) == 3


def test_get_nwb_epoch_returns_raw_record(sweep_with_nwb_epochs):

    record = sweep_with_nwb_epochs.get_nwb_epoch("nwb:E1")
    assert record["start_idx"] == 4
    assert record["end_idx"] == 6
    assert record["tags"]["ShortName"] == "E1"


def test_get_nwb_epoch_unknown_name_raises_key_error(sweep_with_nwb_epochs):

    with pytest.raises(KeyError):
        sweep_with_nwb_epochs.get_nwb_epoch("does_not_exist")


def test_select_epoch_unknown_name_raises_key_error(sweep_with_nwb_epochs):

    with pytest.raises(KeyError):
        sweep_with_nwb_epochs.select_epoch("does_not_exist")


def test_legacy_epoch_selection_still_works_alongside_nwb_epochs(
    sweep_with_nwb_epochs
):

    sweep_with_nwb_epochs.select_epoch("recording")
    assert np.all(
        sweep_with_nwb_epochs.i == [0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2]
    )


def test_nwb_epoch_sharing_a_legacy_epoch_name_does_not_collide(sweep):
    """A nwbEpoch whose ShortName matches a legacy epoch name (e.g.
    "recording") no longer collides -- the "nwb:" prefix on the lookup key
    keeps the two namespaces apart. Both remain independently selectable."""

    same_name_epochs = [_make_nwb_epoch("recording", 0, 3)]

    same_name_sweep = Sweep(
        sweep._t, sweep._v, sweep._i, sampling_rate=sweep.sampling_rate,
        clamp_mode=sweep.clamp_mode, nwb_epochs=same_name_epochs,
    )

    same_name_sweep.select_epoch("recording")
    legacy_i = same_name_sweep.i

    same_name_sweep.select_epoch("nwb:recording")
    nwb_i = same_name_sweep.i

    assert not np.array_equal(legacy_i, nwb_i)
    assert np.all(nwb_i == same_name_sweep._i[0:4])


def test_duplicate_nwb_epoch_short_name_raises(sweep):

    duplicate_epochs = [
        _make_nwb_epoch("E0", 0, 3),
        _make_nwb_epoch("E0", 4, 6),
    ]

    with pytest.raises(ValueError):
        Sweep(
            sweep._t, sweep._v, sweep._i, sampling_rate=sweep.sampling_rate,
            clamp_mode=sweep.clamp_mode, nwb_epochs=duplicate_epochs,
        )
