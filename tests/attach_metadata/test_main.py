""" Tests for the attach_metadata executable
"""

import os
import json
import subprocess as sp
from datetime import datetime

import yaml
import pynwb


def test_cli_dandi_yaml(tmpdir_factory):
    """ Integration tests whether we can write a DANDI-compatible yaml
    """

    tmpdir = str(tmpdir_factory.mktemp("test_cli_dandi_yaml"))
    in_json_path = os.path.join(tmpdir, "input.json")
    out_json_path = os.path.join(tmpdir, "output.json")
    meta_yaml_path = os.path.join(tmpdir, "meta.yaml")

    input_json = {
        "metadata": [
            {
                "name": "species",
                "value": "mouse",
                "sinks": ["dandi_yaml"]
            }
        ],
        "sinks": [
            {
                "name": "dandi_yaml",
                "kind": "DandiYamlSink",
                "targets": [
                    {"stream": meta_yaml_path}
                ]
            }
        ]
    }
    
    with open(in_json_path, "w") as in_json_file:
        json.dump(input_json, in_json_file)

    sp.check_call([
        "python",
        "-m",
        "ipfx.attach_metadata",
        "--input_json",
        in_json_path,
        "--output_json",
        out_json_path
    ])

    with open(out_json_path, "r") as out_json_file:
        out_json = json.load(out_json_file)

    obt_meta_yaml_path = out_json["inputs"]["sinks"][0]["targets"][0]["stream"]
    with open(obt_meta_yaml_path, "r") as obt_meta_yaml_file:
        obt_meta = yaml.load(obt_meta_yaml_file, Loader=yaml.FullLoader)

    assert obt_meta["species"] == "mouse"


def test_cli_nwb2(tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_cli_nwb2"))
    in_json_path = os.path.join(tmpdir, "input.json")
    out_json_path = os.path.join(tmpdir, "output.json")
    in_nwb_path = os.path.join(tmpdir, "input.nwb")
    out_nwb_path = os.path.join(tmpdir, "meta.nwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier='test session',
        session_start_time=datetime.now()
    )
    with pynwb.NWBHDF5IO(path=in_nwb_path, mode="w") as writer:
        writer.write(nwbfile)

    input_json = {
        "metadata": [
            {
                "name": "subject_id",
                "value": "23",
                "sinks": ["nwb2"]
            }
        ],
        "sinks": [
            {
                "name": "nwb2",
                "kind": "Nwb2Sink",
                "config": {"nwb_path": in_nwb_path},
                "targets": [
                    {"output_path": out_nwb_path}
                ]
            }
        ]
    }
    with open(in_json_path, "w") as in_json_file:
        json.dump(input_json, in_json_file)

    sp.check_call([
        "python",
        "-m",
        "ipfx.attach_metadata",
        "--input_json",
        in_json_path,
        "--output_json",
        out_json_path
    ])

    with open(out_json_path, "r") as out_json_file:
        out_json = json.load(out_json_file)
    obt_nwb_path = out_json["inputs"]["sinks"][0]["targets"][0]["output_path"]

    with pynwb.NWBHDF5IO(path=obt_nwb_path, mode="r") as reader:
        obt = reader.read()
        assert obt.subject.subject_id == "23"
