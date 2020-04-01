from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
    Nested, InputFile, OutputFile, Integer, Boolean, String, Float
)


class SweepFeatures(DefaultSchema):
    stimulus_code = String(description="stimulus code", required=True)
    stimulus_name = String(
        description="index of sweep in order of presentation", required=True
    )
    stimulus_amplitude = Float(
        description="amplitude of stimulus", required=True, allow_none=True
    )
    sweep_number = Integer(
        description="index of sweep in order of presentation", required=True
    )
    stimulus_units = String(desription="stimulus units", required=True)
    bridge_balance_mohm = Float(description="bridge balance", allow_none=True)
    pre_vm_mv = Float(allow_none=True)
    leak_pa = Float(allow_none=True)


class CellFeatures(DefaultSchema):
    blowout_mv = Float(description="blash", required=False, allow_none=True)
    seal_gohm = Float(description="blash", allow_none=True)
    electrode_0_pa = Float(description="blash", allow_none=True)
    input_access_resistance_ratio = Float(description="blash", allow_none=True)
    input_resistance_mohm = Float(description="blash", allow_none=True)
    initial_access_resistance_mohm = Float(
        description="blash", allow_none=True
    )


class FxSweepFeatures(SweepFeatures):
    passed = Boolean(description="qc passed or failed", required=True)


class QcSweepFeatures(SweepFeatures):
    pre_noise_rms_mv = Float(description="blah", required=True)
    post_noise_rms_mv = Float(
        description="blah", required=True, allow_none=True
    )
    slow_noise_rms_mv = Float(description="blah", required=True)
    vm_delta_mv = Float(description="blah", required=True, allow_none=True)
    stimulus_duration = Float(description="blah", required=True)
    stimulus_amplitude = Float(
        description="amplitude of stimulus", required=True, allow_none=True
    )


class FeatureExtractionParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    stimulus_ontology_file = InputFile(
        description="stimulus ontology JSON", required=False
    )
    output_nwb_file = OutputFile(description="output nwb file", required=True)
    qc_fig_dir = OutputFile(
        description="output qc figure directory", required=False
    )
    sweep_features = Nested(FxSweepFeatures, many=True)
    cell_features = Nested(CellFeatures, required=True)


class SweepExtractionParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    stimulus_ontology_file = OutputFile(
        description="stimulus ontology JSON", required=False
    )
    manual_seal_gohm = Float(description="blah")
    manual_initial_access_resistance_mohm = Float(description="blah")
    manual_initial_input_mohm = Float(description="blah")


class QcCriteria(DefaultSchema):
    pre_noise_rms_mv_max = Float(description="blash")
    post_noise_rms_mv_max = Float(description="blash")
    slow_noise_rms_mv_max = Float(description="blash")
    vm_delta_mv_max = Float(description="blash")
    blowout_mv_min = Float(description="blash")
    blowout_mv_max = Float(description="blash")
    electrode_0_pa_max = Float(description="blash")
    seal_gohm_min = Float(description="blash")
    input_vs_access_resistance_max = Float(description="blash")
    access_resistance_mohm_min = Float(description="blash")
    access_resistance_mohm_max = Float(description="blash")


class QcParameters(ArgSchema):
    stimulus_ontology_file = InputFile(description="blash", required=False)
    qc_criteria = Nested(QcCriteria, required=True)
    sweep_features = Nested(QcSweepFeatures, many=True, required=True)
    cell_features = Nested(CellFeatures)


class ManualSweepState(DefaultSchema):
    sweep_number = Integer(description="sweep number", required=True)
    passed = Boolean(description="manual override state", required=True)


class PipelineParameters(ArgSchema):
    input_nwb_file = InputFile(description="input nwb file", required=True)
    stimulus_ontology_file = OutputFile(description="blash", required=False)
    output_nwb_file = OutputFile(description="output nwb file", required=True)
    qc_fig_dir = OutputFile(
        description="output qc figure directory", required=False
    )
    qc_criteria = Nested(QcCriteria, required=True)
    manual_sweep_states = Nested(ManualSweepState, required=False, many=True)


class OutputSchema(DefaultSchema):
    input_parameters = Nested(FeatureExtractionParameters,
                              description=("Input parameters the module "
                                           "was run with"),
                              required=True)


class OutputParameters(OutputSchema):
    # Add your output parameters
    pass
