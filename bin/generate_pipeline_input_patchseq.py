import json
import os
import allensdk.internal.core.lims_utilities as lu
import sys

import allensdk.ipfx.ephys_data_set as eds
import allensdk.ipfx.qc_protocol as qcp
import allensdk.core.json_utilities as ju
import os.path

def get_sweep_states_from_lims(specimen_id):

    res = lu.query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    sweep_states = []

    for sweep in res:
        sweep_number = sweep["sweep_number"]

        # only care about manual calls
        if sweep["workflow_state"] == "manual_passed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': True})
        elif sweep["workflow_state"] == "manual_failed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': False})

    return sweep_states

specimen_name = sys.argv[1]

stimulus_ontology_file = eds.DEFAULT_STIMULUS_ONTOLOGY_FILE


d = {}

storage_dir = "/local1/ephys/patchseq/nwb"
cell_dir = os.path.join("/local1/ephys/patchseq/specimens/", specimen_name)

if not os.path.exists(cell_dir):
    os.makedirs(cell_dir)

input_nwb_file = "%s.nwb" % specimen_name

d['input_nwb_file'] = os.path.join(storage_dir, input_nwb_file)
d['output_nwb_file'] = os.path.join(cell_dir, "output.nwb")
d['qc_fig_dir'] = os.path.join(cell_dir,"qc_figs")
d['stimulus_ontology_file'] = stimulus_ontology_file
d['qc_criteria'] = ju.read(qcp.DEFAULT_QC_CRITERIA_FILE)
d['manual_sweep_states'] = []

with open(os.path.join(cell_dir, 'pipeline_input.json'), 'w') as f:
    f.write(json.dumps(d, indent=2))
