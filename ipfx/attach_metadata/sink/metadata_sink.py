"""Base class for metadata sinks, which serve as a staging ground for metadata
attachment.
"""
import abc
import collections
from typing import (
    TypeVar, Union, Collection, List, Dict, Any, Set, Optional
)

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
        if targets is None:
            raise ValueError("must argue targets")

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
