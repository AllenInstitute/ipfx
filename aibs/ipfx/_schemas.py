from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputFile, OutputFile, Integer, Boolean, String, Float

class SweepParameters(DefaultSchema):
    passed = Boolean(description="qc passed or failed", required=True)
    stimulus_code = String(description="stimulus code", required=True)
    stimulus_name = String(description="stimulus name", required=True)
    sweep_number = Integer(description="index of sweep in order of presentation", required=True)
    stimulus_amplitude = Float(description="amplitude of stimulus", required=True, allow_none=True)
    stimulus_units = String(desription="stimulus units", required=True)
    id = Integer(description="id of sweep", allow_none=True)
    bridge_balance_mohm = Float(description="bridge balance", allow_none=True)
    pre_vm_mv = Float(allow_none=True)
    leak_pa = Float(allow_none=True)

class FeatureExtractionParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    output_nwb_file = OutputFile(description="output nwb file", required=True)
    qc_fig_dir = OutputFile(description="output qc figure directory", required=True)
    sweep_list = Nested(SweepParameters, many=True)

class QcParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    known_stimulus_types = String(many=True)

class OutputSchema(DefaultSchema):
    input_parameters = Nested(FeatureExtractionParameters,
                              description=("Input parameters the module "
                                           "was run with"),
                              required=True)

class OutputParameters(OutputSchema):
    # Add your output parameters
    pass
