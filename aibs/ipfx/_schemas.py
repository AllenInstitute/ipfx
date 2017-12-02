from argschema import ArgSchema, ArgSchemaParser
from argschema.schemas import DefaultSchema
from argschema.fields import Nested, InputFile, OutputFile, Integer, Boolean, String, Float

class SweepParameters(DefaultSchema):
    passed = Boolean(description="qc passed or failed", required=True)
    stimulus_type = String(description="type of stimulus", required=True)
    sweep_number = Integer(description="index of sweep in order of presentation", required=True)
    stimulus_amplitude = Float(description="amplitude of stimulus", required=True, allow_none=True)

class FeatureExtractionParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    output_nwb_file = OutputFile(description="output nwb file", required=True)
    qc_fig_dir = OutputFile(description="output qc figure directory", required=True)
    sweep_list = Nested(SweepParameters, many=True)

class OutputSchema(DefaultSchema):
    input_parameters = Nested(FeatureExtractionParameters,
                              description=("Input parameters the module "
                                           "was run with"),
                              required=True)

class OutputParameters(OutputSchema):
    # Add your output parameters
    pass
