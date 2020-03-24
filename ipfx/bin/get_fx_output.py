import ipfx.lims_queries as lq
import glob
import os
import pandas as pd
import argparse

NO_SPECIMEN = "No_specimen_in_LIMS"
NO_OUTPUT_FILE = "No_feature_extraction_output"


def get_fx_output_json(specimen_id):
    """
    Find in LIMS the full path to the json output of the feature extraction module
    If more than one file exists, then chose the latest version

    Parameters
    ----------
    specimen_id

    Returns
    -------
    file_name: string
    """

    query = """
    select err.storage_directory, err.id
    from specimens sp
    join ephys_roi_results err on err.id = sp.ephys_roi_result_id
    where sp.id = %d
    """ % specimen_id

    res = lq.query(query)
    if res:
        err_dir = res[0]["storage_directory"]

        file_list = glob.glob(os.path.join(err_dir, '*EPHYS_FEATURE_EXTRACTION_*_output.json'))
        if file_list:
            latest_file = max(file_list, key=os.path.getctime)   # get the most recent file
            return latest_file
        else:
            return NO_OUTPUT_FILE
    else:
        return NO_SPECIMEN


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
