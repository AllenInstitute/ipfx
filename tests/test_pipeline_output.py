import pytest
import numpy as np
import os
import allensdk.core.json_utilities as ju


import sys
sys.path.append("/local1/pyprojects/allensdk.ipfx/bin") # there is got to be a better way
from run_pipeline import run_pipeline


def check_json_files_are_equal(benchmark_json,test_json):

    benchmark_pipeline_output = ju.read(benchmark_json)
    test_pipeline_output = ju.read(test_json)

    return benchmark_pipeline_output == test_pipeline_output


def test_pipeline_output():
    """
    Validates pipeline output for a list of specimens.
    Runs pipeline, saves data to disk and compares pipeline output to that from the benchmark runs.
    If pipeline output has changed (was not validated), raises assertion error and lists specimens failing validation.


    """
    specimens = np.array([500844783, 509604672, 513876493])

    validation = []

    for specimen in specimens:
        print("testing specimen %s" % specimen)
        benchmark_dir = "/allen/aibs/informatics/module_test_data/ipfx/benchmark_pipeline_output/%s" % specimen
        test_dir = "/allen/aibs/informatics/module_test_data/ipfx/test_pipeline_output/%s" % specimen

        test_pipeline_input_json = os.path.join(test_dir, "pipeline_input.json")
        tpi = ju.read(test_pipeline_input_json)

        test_pipeline_output = run_pipeline(tpi["input_nwb_file"],
                              tpi.get("input_h5_file", None),
                              tpi["output_nwb_file"],
                              tpi["stimulus_ontology_file"],
                              tpi["qc_fig_dir"],
                              tpi["qc_criteria"],
                              tpi["manual_sweep_states"])

        test_pipeline_output_json = os.path.join(test_dir,"pipeline_output.json")
        ju.write(test_pipeline_output_json, test_pipeline_output)

        benchmark_pipeline_output_json = os.path.join(benchmark_dir, "pipeline_output.json")

        val = check_json_files_are_equal(benchmark_pipeline_output_json, test_pipeline_output_json)
        validation.append(val)

    specimens_failing_validation = specimens[~np.array(validation)]

    print("specimens_failing validation: ", specimens_failing_validation)
    assert len(specimens_failing_validation) == 0



def main():
    test_pipeline_output()

if __name__ == "__main__": main()





