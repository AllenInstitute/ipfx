""" Executable for attaching metadata to an ICEPHYS NWB file (and other sinks)
"""

import logging
import copy as cp
from typing import (
    List, Optional, Dict, Sequence, Type
)

from argschema.argschema_parser import ArgSchemaParser

from ipfx.attach_metadata.sink import MetadataSink, default_sink_kinds
from ipfx.attach_metadata._schemas import InputParameters, OutputParameters


def attach_metadata(
        sinks: Dict[str, MetadataSink],
        metadata: List[Dict],
):
    """Attaches metadata inplace to a collection of sinks. Metadata can be 
    provided at the cell and sweep levels.

    Parameters
    ----------
    sinks : Configured sinks for outputs. Will be modified by metadata 
        attachment
    metadata : Each is a dictionary with the following fields

    Raises
    ------
    ValueError : If a piece of metadata is argued to 
        an incompatible sink.

    """

    for metadatum in metadata:
        for sink_name in metadatum["sinks"]:
            sinks[sink_name].register(
                metadatum["name"],
                metadatum["value"],
                metadatum.get("sweep_number", None)
            )


def configure_sinks(
        sink_specs: Sequence[Dict],
        sink_kinds: Optional[Dict[str, Type[MetadataSink]]] = None
) -> Dict[str, MetadataSink]:
    """ Configures metadata sinks based on supplied specifications

    Parameters
    ----------
    sink_specs : Each defines a sink. Must have fields:
        name : str. The unique identifier of this sink
        kind : str. Identifies the class of sink to be configured
        config : dict, optional. Elements are passed as kwargs to sink 
            constructor
        targets : list of dict, technically optional. Elements used to 
            configure targets - the external resources to which sinks will 
            write.
    sink_kinds : Mapping from human-readable names to well-known sink classes.
        Used to initialize sinks based on user input

    Returns
    -------
    Mapping from sink names to configured sinks.

    """

    if sink_kinds is None:
        sink_kinds = default_sink_kinds()

    sinks: Dict[str, MetadataSink] = {}

    for sink_spec in sink_specs:
        kind_class = sink_kinds[sink_spec["kind"]]
        sink = kind_class(**sink_spec.get("config", {}))
        sink.register_targets(sink_spec["targets"])
        sinks[sink_spec["name"]] = sink

    return sinks


def run_attach_metadata(
        sink_specs: Sequence[Dict],
        metadata: List[Dict]
):
    """Attaches metadata to a set of sinks, serializes those sinks to 
    external stores.

    Parameters
    ----------
    sink_configs : Each is used to construct a MetadataSink
    metadata : Will be attached to sinks. Each has
        name : str. Used by the sinks to determine how this value ought to be 
            attached
        value : anything. Attached to the sinks
        sweep_number : int or None. If present, this is sweep metadata and will 
            be attached to the identified sweep.

    Returns
    -------
    lightweight information identifying external stores for sinks
    """

    sinks: Dict[str, MetadataSink] = configure_sinks(sink_specs)
    attach_metadata(sinks, metadata)
    for sink in sinks.values():
        sink.serialize()

    return {
        "sinks": {
            spec["name"]: spec
            for spec in sink_specs
        }
    }


def main():
    """ CLI entry point for attaching metadata to an ICEPHYS NWB file
    """

    parser = ArgSchemaParser(
        schema_type=InputParameters,
        output_schema_type=OutputParameters
    )

    inputs_record = cp.deepcopy(parser.args)
    logging.getLogger().setLevel(inputs_record["log_level"])

    output = {}
    output.update({"inputs": parser.args})
    output.update(
        run_attach_metadata(
            inputs_record["nwb2_sinks"] + inputs_record["dandi_sinks"],
            inputs_record["metadata"]
        )
    )

    parser.output(output)


if __name__ == "__main__":
    main()
