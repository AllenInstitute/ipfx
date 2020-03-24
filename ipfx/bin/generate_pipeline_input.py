import ipfx.qc_feature_evaluator as qcp
import allensdk.core.json_utilities as ju
import os.path
from ipfx.bin.generate_se_input import generate_se_input, parse_args
import ipfx.lims_queries as lq


def generate_pipeline_input(cell_dir=None,
                            specimen_id=None,
                            input_nwb_file=None,
                            plot_figures=False):

    se_input = generate_se_input(cell_dir,
                                 specimen_id=specimen_id,
                                 input_nwb_file=input_nwb_file
                                 )
    pipe_input = dict(se_input)

    if specimen_id:
        pipe_input['manual_sweep_states'] = lq.get_sweep_states(specimen_id)

    elif input_nwb_file:
        pipe_input['manual_sweep_states'] = []

    if plot_figures:
        pipe_input['qc_fig_dir'] = os.path.join(cell_dir,"qc_figs")

    pipe_input['output_nwb_file'] = os.path.join(cell_dir, "output.nwb")
    pipe_input['qc_criteria'] = ju.read(qcp.DEFAULT_QC_CRITERIA_FILE)

    return pipe_input


def main():

    """
    Usage:
    > python generate_pipeline_input.py --specimen_id SPECIMEN_ID --cell_dir CELL_DIR
    > python generate_pipeline_input.py --input_nwb_file INPUT_NWB_FILE --cell_dir CELL_DIR

    """

    kwargs = parse_args()
    pipe_input = generate_pipeline_input(**kwargs)
    cell_dir = kwargs["cell_dir"]

    input_json = os.path.join(cell_dir, 'pipeline_input.json')

    ju.write(input_json, pipe_input)


if __name__ == "__main__": main()



