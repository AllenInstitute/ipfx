"""
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


    def __init__(
            self, 
            nwb_path: PathLike
        ):
        self._targets: List[Dict[str, Any]] = []
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
            file=self._h5_file
        )
        self.nwbfile = self._nwb_io.read()

    def _commit_nwb_changes(self):
        """Write this sink's nwbfile to its _data buffer. After calling this 
        method, further modifications of this sink's NWBFile **WILL NOT** be 
        recorded until _reload_nwbfile
        """

        set_container_sources(self.nwbfile, self._h5_file.filename)
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
                f"expected exactly 1 intracellular electrode, found {len(keys)}"
            )

        return self.nwbfile.ic_electrodes[keys[0]]
    
    def _get_sweep_series(
            self, 
            sweep_id: int
    ) -> pynwb.icephys.PatchClampSeries:
        """ Obtain the PatchClampSeries object corresponding to this sweep id

        Parameters
        ----------
        sweep_id : Unique identifier for this sweep

        Returns
        -------
        A PatchClampSeries object for this sweep

        """
        return self.nwbfile.sweep_table[sweep_id].series.values[0][0]

    def _get_subject(self) -> pynwb.file.Subject:
        """Obtain this NWBFile's subject field, constructing it if needed

        Returns
        -------
        The NWBFile's (potentially newly created) subject field

        """
        if self.nwbfile.subject is None:
            self.nwbfile.subject = pynwb.file.Subject()
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
            raise ValueError(# This is useless
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
        if hasattr(current, "_AbstractContainer__container_source"):
            setattr(
                current, 
                "_AbstractContainer__container_source",
                source
            )

        if hasattr(current, "children"):
            children.extend(current.children)

