import allensdk.core.json_utilities as ju
import os
import make_stimulus_ontology as mso
import lims_queries as lq
import argparse


def generate_se_input(cell_dir,
                      specimen_id=None,
                      input_nwb_file=None,
                      ):

    se_input = dict()

    if specimen_id:
        input_nwb_file = lq.get_input_nwb_file(specimen_id)
        input_h5_file = lq.get_input_h5_file(specimen_id)
        if input_h5_file:
            se_input['input_h5_file'] = input_h5_file

    se_input['input_nwb_file'] = input_nwb_file

    stim_ontology_tags = mso.make_stimulus_ontology_from_lims()
    stim_ontology_json = os.path.join(cell_dir, 'stimulus_ontology.json')

    if not os.path.exists(cell_dir):
        os.makedirs(cell_dir)

    ju.write(stim_ontology_json, stim_ontology_tags)

    se_input["stimulus_ontology_file"] = stim_ontology_json

    return se_input


def parse_args():

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--specimen_id', type=int)
    group.add_argument('--input_nwb_file', type=str)
    parser.add_argument('--cell_dir', type=str, required=True)
    args = parser.parse_args()

    return vars(args)


def main():

    """
    Usage:
    > python generate_se_input.py --specimen_id SPECIMEN_ID --cell_dir CELL_DIR
    > python generate_se_input.py --input_nwb_file input_nwb_file --cell_dir CELL_DIR

    """

    kwargs = parse_args()
    se_input = generate_se_input(**kwargs)

    input_json = os.path.join(kwargs["cell_dir"], 'se_input.json')

    ju.write(input_json, se_input)


if __name__=="__main__": main()
