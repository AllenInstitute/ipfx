from pathlib import Path
from builtins import str
import pytest
import pandas as pd
import os
import allensdk.core.json_utilities as ju
from ipfx.bin.run_pipeline import run_pipeline
from pkg_resources import resource_filename
from dictdiffer import diff


TEST_SPECIMENS_FILE = resource_filename(__name__, 'test_mies_nwb2_specimens.csv')

test_specimens = pd.read_csv(TEST_SPECIMENS_FILE, sep=" ")
test_specimens_params = [tuple(sp) for sp in test_specimens.values]


def rebase(new_base: str, path: str, mkdir: bool = False) ->  str:
    """Utility for joining a path prefix onto a (potentially absolute) path

    Parameters
    ----------
    new_base : Will prefix the output
    path : to be prefixed
    mkdir : if True, make all ancestors of the resulting path

    Returns
    -------
    The joined path, as a string.

    """

    path = Path(path)
    if path.is_absolute:
        path = Path(*path.parts[1:])

    new_path = Path(new_base) / path
    if mkdir:
        new_path.parent.mkdir(parents=True, exist_ok=True)

    return str(new_path)


@pytest.mark.requires_inhouse_data
@pytest.mark.slow
@pytest.mark.parametrize('input_json,output_json', test_specimens_params)
def test_mies_nwb_pipeline_output(input_json, output_json, tmpdir_factory):
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
    print((input_json, output_json))

    pipeline_input = ju.read(input_json)
    test_dir = str(tmpdir_factory.mktemp("test_mies_nwb2_specimens"))

    pipeline_input["output_nwb_file"] = os.path.join(test_dir, "output.nwb")  # Modify path for the test output
    pipeline_input["qc_figs_dir"] = None

    stimulus_ontology_file = pipeline_input.get("stimulus_ontology_file", None)
    if stimulus_ontology_file is not None:
        stimulus_ontology_file = rebase(test_dir, stimulus_ontology_file, True)

    obtained = run_pipeline(pipeline_input["input_nwb_file"],
                                        pipeline_input.get("input_h5_file", None),
                                        pipeline_input["output_nwb_file"],
                                        stimulus_ontology_file,
                                        pipeline_input["qc_figs_dir"],
                                        pipeline_input["qc_criteria"],
                                        pipeline_input["manual_sweep_states"])
    print(type(obtained))
    ju.write(os.path.join(test_dir, 'pipeline_output.json'), obtained)
    obtained = ju.read(os.path.join(test_dir, 'pipeline_output.json'))
    print(type(obtained))
#    assert 0
    expected = ju.read(output_json)

    output_diff = list(diff(expected, obtained, tolerance=0.001))
    if output_diff:
        print(output_diff)
    assert len(output_diff) == 0
