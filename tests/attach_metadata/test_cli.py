""" Tests for the attach_metadata executable
"""
from typing import Dict
import os
import json
import time
import subprocess as sp
from datetime import datetime
import sys

import yaml
import pynwb
import numpy as np
import pytest


class CliRunner:

    def __init__(
            self, 
            tmpdir: str, 
            timeout_seconds: float = sys.float_info.max
    ):
        self.tmpdir = tmpdir
        self.timeout_seconds = timeout_seconds

    def run(self, in_json: Dict) -> Dict:
        in_json_path = os.path.join(self.tmpdir, "input.json")
        with open(in_json_path, "w") as in_json_file:
            json.dump(in_json, in_json_file)

        out_json_path = os.path.join(self.tmpdir, "output.json")
        start_time = time.time()
        sp.check_call([
            "python",
            "-m",
            "ipfx.attach_metadata",
            "--input_json",
            in_json_path,
            "--output_json",
            out_json_path
        ])
        duration = time.time() - start_time
        assert duration < self.timeout_seconds

        with open(out_json_path, "r") as out_json_file:
            return json.load(out_json_file)


def simple_nwb(base_path):
    in_nwb_path = os.path.join(base_path, "input.nwb")
    out_nwb_path = os.path.join(base_path, "meta.nwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    nwbfile.add_acquisition(
      pynwb.TimeSeries(
          name="a timeseries", 
          data=[1, 2, 3], 
          starting_time=0.0, 
          rate=1.0
        )
    )
    with pynwb.NWBHDF5IO(path=in_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    return in_nwb_path, out_nwb_path


@pytest.fixture
def cli_runner(tmpdir_factory, timeout_seconds):
    return CliRunner(
        str(tmpdir_factory.mktemp("cli_test")),
        timeout_seconds
    )


def test_cli_dandi_yaml(cli_runner):
    """ Integration tests whether we can write a DANDI-compatible yaml
    """

    meta_yaml_path = os.path.join(cli_runner.tmpdir, "meta.yaml")

    input_json = {
        "metadata": [
            {
                "name": "species",
                "value": "mouse",
                "sinks": ["dandi_yaml"]
            }
        ],
        "dandi_sinks": [
            {
                "name": "dandi_yaml",
                "targets": [
                    {"output_path": meta_yaml_path}
                ]
            }
        ]
    }
    out_json = cli_runner.run(input_json)

    obt_meta_yaml_path = \
        out_json["sinks"]["dandi_yaml"]["targets"][0]["output_path"]
    with open(obt_meta_yaml_path, "r") as obt_meta_yaml_file:
        obt_meta = yaml.load(obt_meta_yaml_file, Loader=yaml.FullLoader)

    assert obt_meta["species"] == "mouse"


def test_cli_nwb2(cli_runner):
    in_nwb_path, out_nwb_path = simple_nwb(cli_runner.tmpdir)
    input_json = {
        "metadata": [
            {
                "name": "subject_id",
                "value": "23",
                "sinks": ["nwb2"]
            }
        ],
        "nwb2_sinks": [
            {
                "name": "nwb2",
                "config": {"nwb_path": in_nwb_path},
                "targets": [
                    {"output_path": out_nwb_path}
                ]
            }
        ]
    }

    out_json = cli_runner.run(input_json)
    os.remove(in_nwb_path) # make sure we aren't linking
    obt_nwb_path = out_json["sinks"]["nwb2"]["targets"][0]["output_path"]

    with pynwb.NWBHDF5IO(path=obt_nwb_path, mode="r") as reader:
        obt = reader.read()
        assert obt.subject.subject_id == "23"
        assert np.allclose(
            obt.get_acquisition("a timeseries").data[:],
            [1, 2, 3]
        )

def test_cli_multisink(cli_runner):
    in_nwb_path, out_nwb_path = simple_nwb(cli_runner.tmpdir)
    meta_yaml_path = os.path.join(cli_runner.tmpdir, "meta.yaml")

    input_json = {
        "metadata": [
            {
                "name": "subject_id",
                "value": "23",
                "sinks": ["nwb2"]
            },
            {
                "name": "age",
                "value": "56",
                "sinks": ["dandi_yaml"]
            }
        ],
        "nwb2_sinks": [
            {
                "name": "nwb2",
                "config": {"nwb_path": in_nwb_path},
                "targets": [
                    {"output_path": out_nwb_path}
                ]
            }
        ],
        "dandi_sinks": [
            {
                "name": "dandi_yaml",
                "targets": [
                    {"output_path": meta_yaml_path}
                ]
            }
        ]
    }

    out_json = cli_runner.run(input_json)
    os.remove(in_nwb_path)

    obt_nwb_path = out_json["sinks"]["nwb2"]["targets"][0]["output_path"]
    with pynwb.NWBHDF5IO(path=obt_nwb_path, mode="r") as reader:
        obt = reader.read()
        assert obt.subject.subject_id == "23"

    obt_meta_yaml_path = \
        out_json["sinks"]["dandi_yaml"]["targets"][0]["output_path"]
    with open(obt_meta_yaml_path, "r") as obt_meta_yaml_file:
        obt_meta = yaml.load(obt_meta_yaml_file, Loader=yaml.FullLoader)

    assert obt_meta["age"] == "56"