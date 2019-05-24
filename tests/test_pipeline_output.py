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


@pytest.mark.regression
@pytest.mark.parametrize('input_json,output_json', test_specimens_params)
def test_pipeline_output(input_json, output_json, tmpdir_factory):
    """
    Runs pipeline, saves to a json file and compares to the existing pipeline output.
    Raises assertion error if test output is different from the benchmark.

    Parameters
    ----------
    input_json: string json file name of input
    output_json: string json file name of benchmark output
    tmpdir_factory: pytest fixture

    Returns
    -------

    """
    print(input_json, output_json)

    pipeline_input = ju.read(input_json)
    test_dir = str(tmpdir_factory.mktemp("test_specimens"))

    pipeline_input["output_nwb_file"] = os.path.join(test_dir, "output.nwb")  # Modify path for the test output
    pipeline_input["qc_figs_dir"] = os.path.join(test_dir, "qc_figs")

    test_pipeline_output = run_pipeline(pipeline_input["input_nwb_file"],
                                        pipeline_input.get("input_h5_file", None),
                                        pipeline_input["output_nwb_file"],
                                        pipeline_input.get("stimulus_ontology_file", None),
                                        pipeline_input["qc_figs_dir"],
                                        pipeline_input["qc_criteria"],
                                        pipeline_input["manual_sweep_states"])

    ju.write(os.path.join(test_dir, 'pipeline_output.json'), test_pipeline_output)

    test_output = ju.read(os.path.join(test_dir, 'pipeline_output.json'))

    benchmark_output = ju.read(output_json)

    output_diff = list(diff(benchmark_output, test_output, tolerance=0.001))
    if output_diff:
        print(output_diff)
    assert len(output_diff) == 0
