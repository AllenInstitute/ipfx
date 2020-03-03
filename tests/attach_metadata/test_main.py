""" Tests for the attach_metadata executable
"""

import os
import json
import subprocess as sp

import yaml


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
