import os
import allensdk.core.json_utilities as ju
from run_pipeline import assign_sweep_states


"""
    Generate input for the fx module saved in the MODEL_IO_DIR
    qc module run example:
    python run_feature_extraction.py --input_json ../../tests/module_io/fx_input_ivscc.json --output_json ../../tests/module_io/fx_output_ivscc.json

"""
MODULE_IO_DIR = "../../tests/module_io"

se_input_json = os.path.join(MODULE_IO_DIR,'se_input_patchseq.json')
se_output_json = os.path.join(MODULE_IO_DIR,'se_output_patchseq.json')
qc_output_json = os.path.join(MODULE_IO_DIR,"qc_output_patchseq.json")
fx_input_json = os.path.join(MODULE_IO_DIR,"fx_input_patchseq.json")
fx_input_json = os.path.join(MODULE_IO_DIR,"fx_input_patchseq.json")


# se_input_json = os.path.join(MODULE_IO_DIR,'se_input_ivscc.json')
# se_output_json = os.path.join(MODULE_IO_DIR,'se_output_ivscc.json')
# qc_output_json = os.path.join(MODULE_IO_DIR,"qc_output_ivscc.json")
# fx_input_json = os.path.join(MODULE_IO_DIR,"fx_input_ivscc.json")


se_input = ju.read(se_input_json)

se_output = ju.read(se_output_json)

qc_output = ju.read(qc_output_json)



manual_sweep_states = []
assign_sweep_states(manual_sweep_states,
                    qc_output["sweep_states"],
                    se_output["sweep_features"]
                    )

output_nwb_file = "/local1/ephys/module_test_output/output.nwb"
qc_fig_dir = "/local1/ephys/module_test_output/figs"
fx_input_data= {
    'input_nwb_file': se_input['input_nwb_file'],
    'output_nwb_file': output_nwb_file,
    'qc_fig_dir': qc_fig_dir,
    'sweep_features': se_output["sweep_features"],
    'cell_features': se_output['cell_features']
    }

ju.write(fx_input_json, fx_input_data)







