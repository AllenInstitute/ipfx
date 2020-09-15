import allensdk.core.json_utilities as ju
import os.path
from ipfx.bin.run_pipeline import run_pipeline
from ipfx.bin.generate_pipeline_input import generate_pipeline_input
import ipfx.logging_utils as lu
import argparse


def main():
    """
    Convenience script for running ephys pipeline from a given nwb file
    It generates the pipeline input and then calls the run_pipeline executable

    Usage:
    python run_pipeline_from_nwb_file.py <input_nwb_file> <output_dir>
    """

    parser = argparse.ArgumentParser(
        description="Process an nwb file through the ephys pipeline"
    )
    parser.add_argument(
        "input_nwb_file", type=str, help="process this NWB2 file"
    )
    parser.add_argument(
        "output_dir", type=str, help="outputs will be written here"
    )
    parser.add_argument(
        "--qc_criteria_file", type=str, default=None,
        help=(
            "Path to QC criteria json (if not using IPFX defaults)"
        )
    )
    parser.add_argument(
        "--stimulus_ontology_file", type=str, default=None,
        help=(
            "Path to stimulus ontology json (if not using IPFX defaults)"
        )
    )
    parser.add_argument(
        "--write_spikes", type=bool, default=False,
        help="If true will attempt to append spike times to the nwb file",
    )
    parser.add_argument(
        "--input_json", type=str, default="input.json",
        help=(
            "write pipeline input json file here (relative to "
            "OUTPUT_DIR/cell_name, where cell_name is the extensionless "
            "basename of the input NWB file)"
        )
    )
    parser.add_argument(
        "--output_json", type=str, default="output.json", 
        help=(
            "write output json file here (relative to OUTPUT_DIR/cell_name, "
            "where cell_name is the extensionless basename of the input NWB "
            "file)"
        )
    )
    parser.add_argument(
        "--qc_fig_dir", type=str, default=None, const="qc_figs", nargs="?",
        help=(
            "Generate qc figures and store them here (relative to "
            "OUTPUT_DIR/cell_name, where cell_name is the extensionless "
            "basename of the input nwb file). If you supply --qc_fig_dir with " 
            "no arguments, the path will be OUTPUT_DIR/cell_name/qc_figs. If "
            "this argument is not supplied, no figures will be generated."
        )
    )

    args = vars(parser.parse_args())
    output_dir = args["output_dir"]
    input_nwb_file = args["input_nwb_file"]
    input_json = args["input_json"]
    output_json = args["output_json"]

    input_nwb_file_basename = os.path.basename(input_nwb_file)

    cell_name = os.path.splitext(input_nwb_file_basename)[0]
    cell_dir = os.path.join(output_dir, cell_name)
    os.makedirs(cell_dir, exist_ok=True)

    lu.configure_logger(cell_dir)

    pipeline_input = generate_pipeline_input(
        cell_dir=cell_dir,
        input_nwb_file=input_nwb_file,
        plot_figures=args["qc_fig_dir"] is not None,
        qc_fig_dirname=args["qc_fig_dir"],
        qc_criteria_file=args["qc_criteria_file"],
        stimulus_ontology_file=args["stimulus_ontology_file"]
    )

    input_json = os.path.join(cell_dir, input_json)
    ju.write(input_json, pipeline_input)

    #   reading back from disk
    pipeline_input = ju.read(input_json)
    pipeline_output = run_pipeline(pipeline_input["input_nwb_file"],
                                   pipeline_input["output_nwb_file"],
                                   pipeline_input.get("stimulus_ontology_file", None),
                                   pipeline_input.get("qc_fig_dir", None),
                                   pipeline_input["qc_criteria"],
                                   pipeline_input["manual_sweep_states"],
                                   args["write_spikes"],
                                #    if we provide an ontology here, want to use it
                                   update_ontology=False)

    ju.write(os.path.join(cell_dir, output_json), pipeline_output)


if __name__ == "__main__":
    main()



