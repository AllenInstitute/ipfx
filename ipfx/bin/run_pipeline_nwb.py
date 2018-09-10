import json
import sys
import logging
import ipfx.stimulus as stm
import ipfx.qc_protocol as qcp
import allensdk.core.json_utilities as ju
import os.path
from run_pipeline import run_pipeline

stimulus_ontology_file = stm.DEFAULT_STIMULUS_ONTOLOGY_FILE

input_nwb_file = sys.argv[1]
output_dir = sys.argv[2]

input_nwb_file_basename = os.path.basename(input_nwb_file)
specimen_name = os.path.splitext(input_nwb_file_basename)[0]
cell_dir = os.path.join(output_dir, specimen_name)

if not os.path.exists(cell_dir):
    os.makedirs(cell_dir)

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=os.path.join(cell_dir,"log.txt"))
stderrLogger = logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)



def main():

    args = {}
    args['input_nwb_file'] = input_nwb_file
    args['output_nwb_file'] = os.path.join(cell_dir, "output.nwb")
    args['qc_fig_dir'] = os.path.join(cell_dir,"qc_figs")
    args['qc_criteria'] = ju.read(qcp.DEFAULT_QC_CRITERIA_FILE)
    args['manual_sweep_states'] = []

    with open(os.path.join(cell_dir, 'pipeline_input.json'), 'w') as f:
        f.write(json.dumps(args, indent=2))

    output = run_pipeline(args["input_nwb_file"],
                          args.get("input_h5_file", None),
                          args["output_nwb_file"],
                          args.get("stimulus_ontology_file", None),
                          args["qc_fig_dir"],
                          args["qc_criteria"],
                          args["manual_sweep_states"])

    ju.write(os.path.join(cell_dir, "pipeline_output.json"), output)

if __name__ == "__main__": main()



