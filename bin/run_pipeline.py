import argschema as ags
from aibs.ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

def run_pipeline(input_nwb_file, output_nwb_file, stimulus_ontology_file, qc_fig_dir, qc_criteria):
    se_output = run_sweep_extraction(input_nwb_file,
                                     stimulus_ontology_file)
    
    qc_output = run_qc(input_nwb_file,
                       se_output["cell_features"], 
                       se_output["sweep_data"], 
                       qc_criteria)

    sweep_states = { s['sweep_number']:s for s in qc_output['sweep_states'] }
    
    for sweep in se_output["sweep_data"]:
        sweep['passed'] = sweep_states[sweep['sweep_number']]['passed']

    fx_output = run_feature_extraction(input_nwb_file,
                                       output_nwb_file,
                                       qc_fig_dir,
                                       se_output['sweep_data'],
                                       se_output['cell_features'])

    return fx_output

def main():
    module = ags.ArgSchemaParser(schema_type=PipelineParameters)    
    run_pipeline(module.args["input_nwb_file"],
                 module.args["output_nwb_file"],
                 module.args["stimulus_ontology_file"],
                 modlue.args["qc_fig_dir"],
                 module.args["qc_criteria"])

if __name__ == "__main__": main()
