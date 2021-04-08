import pytest
import os
from ipfx.qc_feature_extractor import filter_smoketests, minimum_e0


def test_filter_smoketests():
    smoke_test_sweeps = [0,1,10]
    cell_attached_sweep = 2
    candidates = filter_smoketests(smoke_test_sweeps, cell_attached_sweep)
    assert candidates == [0,1]


def test_minimum_e0_empty():
    e0_candidates = []
    min_e0 = minimum_e0(e0_candidates)
    assert min_e0 == None


def test_minimum_e0_negative():
    e0_candidates = [-150, 50, -35, 120]
    min_e0 = minimum_e0(e0_candidates)
    assert min_e0 == -35


def test_minimum_e0_positive():
    e0_candidates = [-150, 50, -85, 120]
    min_e0 = minimum_e0(e0_candidates)
    assert min_e0 == 50
