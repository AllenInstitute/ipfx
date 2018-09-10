import pytest
import pandas as pd
import os
import allensdk.core.json_utilities as ju
from ipfx.bin.run_pipeline import run_pipeline
from pkg_resources import resource_filename
from dictdiffer import diff


if 'TEST_SPECIMENS_FILE' in os.environ:
    TEST_SPECIMENS_FILE = os.environ['TEST_SPECIMENS_FILE']
else:
    TEST_SPECIMENS_FILE = resource_filename(__name__, 'test_specimens.csv')

test_specimens = pd.read_csv(TEST_SPECIMENS_FILE, sep=" ")
test_specimens_params = [tuple(sp) for sp in test_specimens.values]


def check_json_files_are_equal(json1, json2):
    """
    Compare dicts read from json

    Parameters
    ----------
    json1
    json2

    Returns
    -------
    bool: True if equal
    """

    d1 = ju.read(json1)
    d2 = ju.read(json2)

    return d1 == d2




@pytest.mark.parametrize('specimen,benchmark_pipeline_input_json,benchmark_pipeline_output_json', test_specimens_params)
def test_pipeline_output(specimen, benchmark_pipeline_input_json, benchmark_pipeline_output_json, tmpdir_factory):
    """
    Runs pipeline, saves to a json file and compares to the existing pipeline output.
    Raises assertion error if the dicts with pipeline outputs are not identical.

    Parameters
    ----------
    specimen: string/int specimen name/id
    benchmark_pipeline_input_json: string json file name
    benchmark_pipeline_output_json: string json file name
    tmpdir_factory: pytest fixture

    Returns
    -------

    """

    print (specimen,benchmark_pipeline_input_json,benchmark_pipeline_output_json)

    pipeline_input = ju.read(benchmark_pipeline_input_json)
    test_dir = str(tmpdir_factory.mktemp("%s" % specimen))

    pipeline_input["output_nwb_file"] = os.path.join(test_dir,"output.nwb")  # Modify path for the test output
    pipeline_input["qc_figs_dir"] = os.path.join(test_dir,"qc_figs")

    test_pipeline_output = run_pipeline(pipeline_input["input_nwb_file"],
                                        pipeline_input.get("input_h5_file", None),
                                        pipeline_input["output_nwb_file"],
                                        pipeline_input.get("stimulus_ontology_file", None),
                                        pipeline_input["qc_figs_dir"],
                                        pipeline_input["qc_criteria"],
                                        pipeline_input["manual_sweep_states"])

    test_pipeline_output_json = os.path.join(test_dir,'pipeline_output.json')
    test_pipeline_input_json = os.path.join(test_dir,'pipeline_input.json')

    ju.write(test_pipeline_output_json, test_pipeline_output)
    ju.write(test_pipeline_input_json, pipeline_input)

    d1 = ju.read(benchmark_pipeline_output_json)
    d2 = ju.read(test_pipeline_output_json)

    result = list(diff(d1, d2, tolerance=0.001))
    print result
    assert len(result) == 0

