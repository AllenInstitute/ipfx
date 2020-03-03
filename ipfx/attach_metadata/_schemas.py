""" Input and output schemas for the attach_metadata executable
"""

from argschema.schemas import DefaultSchema, ArgSchema
from argschema.fields import Nested, String, Dict, Field, List, Int


class SinkSpecification(DefaultSchema):
    """ Fully specify a sink (reusable output destination)
    """

    name = String(
        description="Identifier for this sink. Should be unique",
        required=True
    )
    kind = String(
        description="What sort of sink is this?",
        required=True
    )
    config = Dict(
        description="Parameters required to define this sink",
        required=True,
        default=dict()
    )
    targets = List(
        Dict,
        description=(
            "Each entry defines a new target to which this sink will be written"
        ),
        required=True,
        default=list()
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
    sinks = Nested(
        SinkSpecification, 
        description="specify outputs to which metadata will be written", 
        many=True,
        required=True
    )


class OutputParameters(DefaultSchema):
    """ Outputs produced by attach_metadata
    """

    inputs = Nested(
        InputParameters, 
        description="The parameters argued to this executable",
        required=True
    )
