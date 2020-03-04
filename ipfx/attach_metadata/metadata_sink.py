""" Exposes a collection of configurable, reusable sinks for ICEPHYS metadata
"""

import abc
import io
import copy as cp
import collections
from typing import (
    Any, Type, Dict, Optional, TypeVar, Collection, Union, Set, List,
)

import yaml
import pynwb


TV = TypeVar("TV")
OneOrMany = Union[TV, Collection[TV]]


class MetadataSink(abc.ABC):
    """ Abstract(ish) interface for metadata sinks.
    """

    @abc.abstractproperty
    def targets(self) -> List[Dict[str, Any]]:
        """ A sequence of preregistered targets. Calling serialize with no
        arguments will write to these.
        """

    @abc.abstractproperty
    def supported_cell_fields(self) -> Set[str]:
        """ The names of each cell-level field supported by this sink.
        """

    @abc.abstractproperty
    def supported_sweep_fields(self) -> Set[str]:
        """ The names of each sweep-level field supported by this sink.
        """

    @abc.abstractmethod
    def serialize(self, targets: Optional[OneOrMany[Dict[str, Any]]] = None):
        """ Writes this sink's data to an external target or targets. Does not 
        modify this sink.

        Parameters
        ----------
        targets : If provided, these targets will be written to. Otherwise, 
            write to targets previously defined by register_target.
        """

    @abc.abstractmethod
    def register(self, name: str, value: Any, sweep_id: Optional[int] = None):
        """ Attaches a named piece of metadata to this sink's internal store. 
        Should dispatch to a protected method which carries out appropriate 
        validations and transformations.

        Parameters
        ----------
        name : the well-known name of the metadata
        value : the value of the metadata (before any required transformations)
        sweep_id : If provided, this will be interpreted as sweep-level 
            metadata and sweep_id will be used to identify the sweep to which 
            value ought to be attached. If None, this will be interpreted as 
            cell-level metadata

        Raises
        ------
        ValueError : An argued piece of metadata is not supported by this sink
        """

    def register_targets(self, targets: OneOrMany[Dict[str, Any]]):
        """ Configures targets for this sink. Calls to serialize (without 
        new targets supplied) will write to these targets.

        Parameters
        ----------
        targets : Configure these targets. Each should be a dictionary which 
            passes this class's validate_target method

        """
        targets = self._ensure_plural_targets(targets)
        for target in targets:
            self.register_target(target)

    def register_target(self, target: Dict[str, Any]):
        """Preregister a single target specification on this sink.

        Parameters
        ----------
        target : Must provide parameters required for serialization

        """
        self.targets.append(target)

    def _ensure_plural_targets(
            self, 
            targets: Optional[OneOrMany[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Convenience for getting a list of targets. Used in register_targets.
        
        Parameters
        ----------
        targets : 
            if None: use this objects targets attribute
            if a single dict: wrap it in a list
            if a sequence: convert it to a list

        Returns
        -------
        a list of target dictionaries

        """

        if targets is None:
            targets = self.targets
        elif isinstance(targets, dict):
            targets = [targets]
        elif isinstance(targets, collections.Sequence):
            targets = list(targets)
        else:
            raise ValueError(
                f"unable to serialize to targets of type {type(targets)}"
            )

        return targets


class DandiYamlSink(MetadataSink):
    """ Sink specialized for writing data to a DANDI-compatible YAML file.
    """

    @property
    def targets(self) -> List[Dict[str, Any]]:
        return self._targets

    @property
    def supported_cell_fields(self) -> Set[str]:
        return {
            "species",
            "age",
            "sex",
            "gender",
            "date_of_birth",
            "genotype",
            "cre_line"
        }

    @property
    def supported_sweep_fields(self) -> Set[str]:
        return set()

    def __init__(self):

        self._targets: List[Dict] = []
        self._data = {}

    def serialize(self, targets: Optional[OneOrMany[Dict[str, Any]]] = None):
        """ Write this sink's data to one or more 
        """

        for target in self._ensure_plural_targets(targets):
            target = cp.deepcopy(target)
            if not isinstance(target["stream"], io.IOBase):
                target["stream"] = open(target["stream"], "w")

            yaml.dump(self._data, **target)
            target["stream"].close()

    def register(
            self, 
            name: str, 
            value: Any, 
            sweep_id: Optional[int] = None
    ):
        """ Attaches a named piece of metadata to this sink's internal store. 
        Should dispatch to a protected method which carries out appropriate 
        validations and transformations.

        Parameters
        ----------
        name : the well-known name of the metadata
        value : the value of the metadata (before any required transformations)
        sweep_id : If provided, this will be interpreted as sweep-level 
            metadata and sweep_id will be used to identify the sweep to which 
            value ought to be attached. If None, this will be interpreted as 
            cell-level metadata

        Raises
        ------
        ValueError : An argued piece of metadata is not supported by this sink
        """

        if name in self.supported_cell_fields:
            # this format is just a straightforward mapping
            self._data[name] = value

        else:
            raise ValueError(
                f"don't know how to attach metadata field: {name}\n"
            )

class Nwb2Sink(MetadataSink):
    """ A metadata sink which modifies an NWBFile (in-memory representation of 
    an NWB 2 file)
    """

    @property
    def targets(self) -> List[Dict[str, Any]]:
        return self._targets


    @property
    def supported_cell_fields(self) -> Set[str]:
        return {
            # general:

            "subject_id",
            # "specimen_id", # TODO: currently unsupported by NWB
            # "citation_policy", # TODO: currently unsupported by NWB
            "institution",
            # "external_solution_recipe", # TODO: currently unsupported by NWB
            # "recording_temperature", # TODO: currently unsupported by NWB
            # "reporter_status", # TODO: currently unsupported by NWB
            "electrode_id",
            "electrode_resistance",
            # "electrode_internal_solution_recipe", # TODO: currently unsupported by NWB
        }

    @property
    def supported_sweep_fields(self) -> Set[str]:
        return {
            # general

            "gain",
            # "output_low_pass_filter_type", # TODO: NWB only has a "filtering" string
            # "output_low_pass_filter_cutoff_frequency" # TODO: NWB only has a "filtering" string
            # "output_high_pass_filter_type", # TODO: NWB only has a "filtering" string
            # "output_high_pass_filter_cutoff_frequency", # TODO: NWB only has a "filtering" string
            # "holding" # TODO: appropriate NWB field not listed

            # specific to current-clamp sweeps:

            "bridge_balance_enabled",
            "fast_capacitance_compensation_enabled",
            # "leak_current_enabled", # TODO: currently unsupported by NWB
            # "leak_current_value", # TODO: currently unsupported by NWB

            # specific to voltage-clamp sweeps:

            "whole_cell_capacitance_compensation_enabled", # TODO: currently unsupported by NWB
            "series_resistance_correction_enabled",
        }


    def __init__(self, nwbfile: pynwb.NWBFile, copy_file=True):
        self._targets: List[Dict[str, Any]] = []
        if copy_file:
            nwbfile = cp.deepcopy(nwbfile)
        self.nwbfile = nwbfile

    def _single_ic_electrode(self) -> pynwb.icephys.IntracellularElectrode:
        """Find the unique electrode used during this session.

        Returns
        -------
        electrode object

        Raises
        ------
        ValueError : If there is not exactly 1 intracellular electrode in this 
            file.

        """

        keys = list(self.nwbfile.ic_electrodes.keys())
        
        if len(keys) != 1:
            raise ValueError(
                f"expected exactly 1 intracellular electrode, found {len(keys)}"
            )

        return self.nwbfile.ic_electrodes[keys[0]]
    
    def _get_sweep_series(self, sweep_id: int):
        """
        """
        return self.nwbfile.sweep_table[sweep_id].series.values[0][0]

    def register(self, name: str, value: Any, sweep_id: Optional[int] = None):
        """ Attaches a named piece of metadata to this sink's internal store. 
        Should dispatch to a protected method which carries out appropriate 
        validations and transformations.

        Parameters
        ----------
        name : the well-known name of the metadata
        value : the value of the metadata (before any required transformations)
        sweep_id : If provided, this will be interpreted as sweep-level 
            metadata and sweep_id will be used to identify the sweep to which 
            value ought to be attached. If None, this will be interpreted as 
            cell-level metadata

        Raises
        ------
        ValueError : An argued piece of metadata is not supported by this sink
        """

        if sweep_id is None:
            if name == "subject_id":
                self.nwbfile.subject.subject_id = value
            elif name == "institution":
                self.nwbfile.institution = value
            elif name == "electrode_id":
                self._single_ic_electrode().name = str(value)
            elif name == "electrode_resistance":
                self._single_ic_electrode().resistance = value

        elif isinstance(sweep_id, int):
            series = self._get_sweep_series(sweep_id)

            if name == "gain":
                series.gain = value
            elif name == "bridge_balance_enabled":
                series.bridge_balance = value # TODO: this is a float, but named enabled? I think we might need to read this from nwb -> other store
            elif name == "fast_capacitance_compensation_enabled":
                series.capacitance_compensation = value
            elif name == "whole_cell_capacitance_compensation_enabled":
                series.whole_cell_capacitance_comp = value
            elif name == "series_resistance_correction_enabled":
                series.resistance_comp_correction = value

        else:
            raise ValueError(
                "unable to attach metadata field: "
                f"{name} (sweep_id: {sweep_id})"
            )


def default_sink_kinds() -> Dict[str, Type[MetadataSink]]:
    """ Maps string names to metadata sink classes
    """
    return {
        "DandiYamlSink": DandiYamlSink
    }
