import pytest
import pandas as pd
import os
import allensdk.core.json_utilities as ju
from allensdk.ipfx.bin.run_pipeline import run_pipeline
from pkg_resources import resource_filename

if 'TEST_SPECIMENS_FILE' in os.environ:
    TEST_SPECIMENS_FILE = os.environ['TEST_SPECIMENS_FILE']
else:
    TEST_SPECIMENS_FILE = resource_filename(__name__, 'test_specimens.csv')

test_specimens = pd.read_csv(TEST_SPECIMENS_FILE, sep=" ")
test_specimens_params = [tuple(sp) for sp in test_specimens.values]


def check_json_files_are_equal(benchmark_json,test_json):

    benchmark_pipeline_output = ju.read(benchmark_json)
    test_pipeline_output = ju.read(test_json)

    return benchmark_pipeline_output == test_pipeline_output


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

    bpi = ju.read(benchmark_pipeline_input_json)
    test_dir = str(tmpdir_factory.mktemp("%s" % specimen))

    tpi = dict(bpi)
    tpi["output_nwb_file"] = os.path.join(test_dir,"output.nwb")
    tpi["qc_figs_dir"] = os.path.join(test_dir,"qc_figs")

    test_pipeline_output = run_pipeline(tpi["input_nwb_file"],
                          tpi.get("input_h5_file", None),
                          tpi["output_nwb_file"],
                          tpi["stimulus_ontology_file"],
                          tpi["qc_figs_dir"],
                          tpi["qc_criteria"],
                          tpi["manual_sweep_states"])

    test_pipeline_output_json = os.path.join(test_dir,'pipeline_output.json')
    test_pipeline_input_json = os.path.join(test_dir,'pipeline_input.json')

    ju.write(test_pipeline_output_json, test_pipeline_output)
    ju.write(test_pipeline_input_json, tpi)

    assert check_json_files_are_equal(benchmark_pipeline_output_json, test_pipeline_output_json)

