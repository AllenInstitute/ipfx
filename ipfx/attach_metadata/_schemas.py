""" Input and output schemas for the attach_metadata executable
"""

from argschema.schemas import DefaultSchema, ArgSchema
from argschema.fields import (
    Nested, String, Dict, Field, List, Int, InputFile, OutputFile
)


class SinkSpecification(DefaultSchema):
    """Basic requirements for sinks
    """

    name = String(
        description="Identifier for this sink. Should be unique",
        required=True
    )


class Nwb2SinkConfig(DefaultSchema):
    """Configure an Nwb2Sink (by arguing the input nwb path)
    """
    nwb_path = InputFile(
        description=(
            "Path to input NWB. This will serve as the basis for the output "
            "file."
        ),
        required=True
    )


class Nwb2SinkTarget(DefaultSchema):
    """Configure an output target for an Nwb2 Sink
    """
    output_path = OutputFile(
        description=(
            "Output path to which file with attached metadata will be written"
        ),
        required=True
    )


class Nwb2SinkSpecification(SinkSpecification):
    """Specify options for writing / adding metadata to an NWB2 file
    """
    kind = String(
        description="what sort of sink is this?",
        required=True,
        default="Nwb2Sink"
    )
    config = Nested(
        Nwb2SinkConfig,
        description="Parameters required to define this sink",
        required=True,
        many=False
    )
    targets = Nested(
        Nwb2SinkTarget,
        description="Targets (output nwb files) which will be written to",
        required=True,
        many=True
    )


class DandiSinkTarget(DefaultSchema):
    """Specify an output target for a DANDI metadata sink
    """
    output_path = OutputFile(
        description=(
            "Outputs will be written here. Currently only yaml is "
            "supported"
        )
    )


class DandiSinkSpecification(SinkSpecification):
    """Specify a sink which writes data to a DANDI-compatible format
    """
    kind = String(
        description="what sort of sink is this?",
        required=True,
        default="DandiSink"
    )
    targets = Nested(
        Nwb2SinkTarget,
        description="Targets (currently yaml files) which will be written to",
        required=True,
        many=True
    )


class Metadatum(DefaultSchema):
    """ A piece of lightweight data
    """

    name = String(
        description=(
            "Identifier for this piece of metadata. Sinks will use this field "
            "in order to determine how metadata ought to be stored."
        ),
        required=True
    )
    value = Field(
        description="The value of this metadata",
        required=True
    )
    sweep_number = Int(
        description="If this is a ",
        required=False
    )
    sinks = List(
        String,
        description="Sink(s) to which this metadatum ought to be written",
        required=True,
        default=list,
        validate=lambda x: len(x) > 0
    )


class InputParameters(ArgSchema):
    """ Inputs required by attach_metadata
    """

    metadata = Nested(
        Metadatum,
        description=(
            "A piece of metadata, which will be written to 1 or more sinks"
        ),
        many=True,
        required=True
    )
    nwb2_sinks = Nested(
        Nwb2SinkSpecification,
        description="Specify nwb files to which metadata should be attached",
        many=True,
        default=[]
    )
    dandi_sinks = Nested(
        DandiSinkSpecification,
        description=(
            "Specify dandi-compatible files to which metadata should be "
            "attached"
        ),
        many=True,
        default=[]
    )


class OutputParameters(DefaultSchema):
    """ Outputs produced by attach_metadata
    """

    inputs = Nested(
        InputParameters, 
        description="The parameters argued to this executable",
        required=True
    )
    sinks = Dict(
        description="The sinks to which metadata was attached",
        required=True,
        many=True
    )
