"""
"""
from unittest.mock import patch

import pytest
import numpy as np

from ipfx.attach_metadata.sink import metadata_sink


@pytest.mark.parametrize("inp, expt", [
    [{}, [{}]],
    [[{}, {}], [{}, {}]],
    [None, "fizz"]
])
def test_ensure_plural_targets(inp, expt):
    with patch.multiple(
        metadata_sink.MetadataSink,
        __abstractmethods__=[]
    ):

        class WithTargets(metadata_sink.MetadataSink):
            @property
            def targets(self):
                return "fizz"

        sink = WithTargets()
        obt = sink._ensure_plural_targets(inp)
        assert len(obt) == len(expt)
        assert type(obt) == type(expt)


def test_register_target():
    with patch.multiple(
        metadata_sink.MetadataSink,
        __abstractmethods__=[]
    ):

        class WithTargets(metadata_sink.MetadataSink):
            @property
            def targets(self):
                if not hasattr(self, "_targets"):
                    self._targets = []
                return self._targets

        sink = WithTargets()
        sink.register_target(1)
        sink.register_target(2)
        assert np.allclose(sink.targets, [1, 2])


def test_register_targets():
    with patch.multiple(
        metadata_sink.MetadataSink,
        __abstractmethods__=[]
    ):

        class WithTargets(metadata_sink.MetadataSink):
            @property
            def targets(self):
                if not hasattr(self, "_targets"):
                    self._targets = []
                return self._targets

        sink = WithTargets()
        sink.register_targets([1, 2, 3])
        assert np.allclose(sink.targets, [1, 2, 3])
