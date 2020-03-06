"""Sinks (in-memory staging areas) for metadata 
"""

from typing import Dict, Type

from ipfx.attach_metadata.sink.metadata_sink import MetadataSink
from ipfx.attach_metadata.sink.dandi_yaml_sink import DandiYamlSink
from ipfx.attach_metadata.sink.nwb2_sink import Nwb2Sink


def default_sink_kinds() -> Dict[str, Type[MetadataSink]]:
    """ Maps string names to metadata sink classes
    """
    return {
        "DandiSink": DandiYamlSink,
        "Nwb2Sink": Nwb2Sink
    }
