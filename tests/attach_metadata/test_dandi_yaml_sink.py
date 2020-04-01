"""
"""
import os

import yaml
import pytest

from ipfx.attach_metadata.sink import DandiYamlSink


def test_serialize(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_serialize_yaml.py"))
    yaml_path = os.path.join(tmpdir, "test.yaml")

    sink = DandiYamlSink()
    sink._data = {"a": 1}
    sink.serialize({"output_path": yaml_path})

    with open(yaml_path, "r") as yaml_file:
        obt = yaml.load(yaml_file, Loader=yaml.FullLoader)
        assert obt["a"] == 1


@pytest.mark.parametrize("name, value", [
    ["species", "mouse"],
    ["age", "56"],
    ["sex", "M"],
    ["gender", "F"],
    ["date_of_birth", "3/5/2020"],
    ["genotype", "wt"],
    ["cre_line", "wt"],

])
def test_register(name, value):

    sink = DandiYamlSink()
    sink.register(name, value)
    assert sink._data[name] == value