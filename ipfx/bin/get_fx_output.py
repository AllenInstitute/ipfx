from __future__ import absolute_import
from ipfx.lims_queries import get_fx_output_json
import pandas as pd
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    return vars(args)


def main():
    """
    Usage:
    $python get_fx_output.py --input_file IN_FILE --output_file OUT_FILE
    IN_FILE: name of the input file including a single column with the header 'specimen_id'
    OUT_FILE: name of the output file that includes columns 'specimen_id' and 'fx_output_json'
    """
    kwargs = parse_args()
    specimen_file = pd.read_csv(kwargs["input_file"], sep=" ")
    specimen_ids = specimen_file["specimen_id"].values

    fx_out = [
        {"specimen_id": specimen_id, "fx_output_json": get_fx_output_json(specimen_id)}
        for specimen_id in specimen_ids]

    fx_out_df = pd.DataFrame(fx_out)
    fx_out_df.to_csv(kwargs["output_file"], sep=" ", na_rep="NA")


if __name__=="__main__":
    main()
