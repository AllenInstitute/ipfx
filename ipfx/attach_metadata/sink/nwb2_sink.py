"""Sink for appending to an NWBFile (not in place)
"""
import io
from pathlib import Path
from typing import (
    List, Dict, Any, Set, Optional, Union
)

import pynwb
import h5py
import hdmf

from ipfx.attach_metadata.sink.metadata_sink import (
    MetadataSink, OneOrMany
)


PathLike = Union[
    str, 
    Path
]


class Nwb2Sink(MetadataSink):
    """ A metadata sink which modifies an NWBFile (in-memory representation of 
    an NWB 2 file)
    """

    @property
    def targets(self) -> List[Dict[str, Any]]:
        return self._targets

    @property
    def supported_cell_fields(self) -> Set[str]:
        return {  # TODO update list based on dandi reqs
            "subject_id",
            "institution",
            "electrode_id",
            "electrode_resistance",
        }

    @property
    def supported_sweep_fields(self) -> Set[str]:
        return {  # TODO update list based on dandi reqs
            "gain",
        }

    def __init__(
            self, 
            nwb_path: Optional[PathLike]
    ):
        self._targets: List[Dict[str, Any]] = []
        if nwb_path is not None:
            self._initial_load_nwbfile(nwb_path)

    def _initial_load_nwbfile(self, nwb_path: PathLike):
        """Reads an nwbfile from an argued path into memory

        Parameters
        ----------
        nwb_path : points to an h5 nwb file.

        """

        with open(nwb_path, "rb") as file_:
            self._data = io.BytesIO(file_.read())
        self._reload_nwbfile()

    def _reload_nwbfile(self):
        """ Construct an nwbfile from this object's _data buffer.
        """

        self._h5_file = h5py.File(self._data, "r+")
        self._nwb_io = pynwb.NWBHDF5IO(
            path=self._h5_file.filename,
            mode="r+",
            file=self._h5_file,
            load_namespaces=True
        )
        self.nwbfile = self._nwb_io.read()

    def _commit_nwb_changes(self):
        """Write this sink's nwbfile to its _data buffer. After calling this 
        method, further modifications of this sink's NWBFile **WILL NOT** be 
        recorded until _reload_nwbfile
        """

        set_container_sources(self.nwbfile, self._h5_file.filename)
        self.nwbfile.set_modified(True)
        self._nwb_io.write(self.nwbfile)
        self._nwb_io.close()
        self._h5_file.close()

    def _get_single_ic_electrode(self) -> pynwb.icephys.IntracellularElectrode:
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
                "expected exactly 1 intracellular electrode, found "
                f"{len(keys)}"
            )

        electrode = self.nwbfile.icephys_electrodes[keys[0]]
        electrode.set_modified(True)

        return electrode
    
    def _get_sweep_series(
            self, 
            sweep_id: int
    ) -> List[pynwb.icephys.PatchClampSeries]:
        """ Obtain the PatchClampSeries object corresponding to this sweep id

        Parameters
        ----------
        sweep_id : Unique identifier for this sweep

        Returns
        -------
        A collection of PatchClampSeries object for this sweep

        """
        return self.nwbfile.sweep_table.get_series(sweep_id)
        
    def _get_subject(self) -> pynwb.file.Subject:
        """Obtain this NWBFile's subject field, constructing it if needed

        Returns
        -------
        The NWBFile's (potentially newly created) subject field

        """
        if self.nwbfile.subject is None:
            self.nwbfile.subject = pynwb.file.Subject()
        self.nwbfile.subject.set_modified(True)
        return self.nwbfile.subject

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
                self._get_subject().subject_id = value
            elif name == "institution":
                self.nwbfile.institution = value
            elif name == "electrode_id":
                self._get_single_ic_electrode().name = str(value)
            elif name == "electrode_resistance":
                self._get_single_ic_electrode().resistance = value
            else:
                self._cant_attach(name, sweep_id)

        elif isinstance(sweep_id, int):
            all_series = self._get_sweep_series(sweep_id)
            for series in all_series:
                if name == "gain":
                    series.gain = value
                else:
                    self._cant_attach(name, sweep_id)
        else:
            self._cant_attach(name, sweep_id)

    def _cant_attach(self, name: str, sweep_id: Optional[int]):
        """Helper - raises if attachment of a particular field is not supported
        """
        raise ValueError(
            "unable to attach metadata field: "
            f"{name} (sweep_id: {sweep_id})"
        )

    def serialize(self, targets: Optional[OneOrMany[Dict[str, Any]]] = None):
        """ Writes this sink's data to an external target or targets. Does not 
        modify this sink.

        Parameters
        ----------
        targets : If provided, these targets will be written to. Otherwise, 
            write to targets previously defined by register_target.
        """
        self._commit_nwb_changes()

        for target in self._ensure_plural_targets(targets):
            with open(target["output_path"], "wb") as file_:
                file_.write(self._data.getvalue())

        self._reload_nwbfile()


def set_container_sources(
        container: hdmf.container.AbstractContainer,
        source: str
):
    """Traverse an NWBFile starting at a given container, setting the
    container_source attribute inplace on each container.

    Parameters
    ----------
    container : container_source will be set on this object as well as on 
        each of its applicable children.
    source : The new value of container source
    """
    children = [container]
    while children:
        current = children.pop()

        # 💀💀💀
        # container_source is set on write, but cannot be overrwritten, making 
        # read -> modify -> write elsewhere
        # pretty tricky!
        # this is a fragile workaround
        if hasattr(current, "_AbstractContainer__container_source"):
            setattr(
                current, 
                "_AbstractContainer__container_source",
                source
            )

        if hasattr(current, "children"):
            children.extend(current.children)
