import argschema as ags
from aibs.ipfx._schemas import PipelineParameters

from run_sweep_extraction import run_sweep_extraction
from run_qc import run_qc
from run_feature_extraction import run_feature_extraction

def main():
    module = ags.ArgSchemaParser(schema_type=PipelineParameters)    
    se_output = run_sweep_extraction(module.args["input_nwb_file"],
                                     module.args["stimulus_ontology_file"])
    
    qc_output = run_qc(module.args["input_nwb_file"],
                       se_output["cell_features"], 
                       se_output["sweep_data"], 
                       module.args["qc_criteria"])

    sweep_states = { s['sweep_number']:s for s in qc_output['sweep_states'] }
    
    for sweep in se_output["sweep_data"]:
        sweep['passed'] = sweep_states[sweep['sweep_number']]['passed']

    fx_output = run_feature_extraction(module.args["input_nwb_file"],
                                       module.args["output_nwb_file"],
                                       module.args["qc_fig_dir"],
                                       se_output['sweep_data'],
                                       se_output['cell_features'])
                                       
                       

    

if __name__ == "__main__": main()
