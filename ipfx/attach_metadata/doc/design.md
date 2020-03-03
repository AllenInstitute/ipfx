Attach Metadata
===============

`attach_metadata` is a standalone executable which 
1. reads an NWB2 file containing processed, feature-extracted data from an icephys experiment.
1. dispatches argued cell- and sweep-level metadata to appropriate sinks. This will be a mix of
    1. An in-memory representation of an NWB2 file
    1. An in-memory representation of a generic mapping
1. serializes these sinks to
    1. an on-disk NWB2 file
    1. an on-disk yaml

This executable is the last stage within a pipeline which incrementally builds DANDI-compatible NWB2 files. This pipeline is deployed in LIMS2 - the previous version of this module runs on the `EPHYS_PUBLISH_NWB_V2_QUEUE` in LIMS2, but only supports NWB1.

Some handy links:
- [v2 (NWB1 version)](http://confluence.corp.alleninstitute.org/pages/viewpage.action?spaceKey=IT&title=IVSCC+NWB+publish+module%2C+v2)
- [current version confluence](http://confluence.corp.alleninstitute.org/pages/viewpage.action?spaceKey=PP&title=Publishing+IVSCC+NWB2+publish+module%2C+v3)
- [NWB2 DANDI icephys metadata list](https://docs.google.com/document/d/1KaQTtZ1AWSjHQsC3XO4k3BT5Z_pjVb4F6fIzrTEClhs/edit#heading=h.diqkrf5s0jti)
- [NWB2 icephys extension proposal](https://docs.google.com/document/d/1cAgsXv26BmQoVfa7Greyxs0oc4IGH-t5aJsm-AwUAAE/edit)

goals
-----
1. NWB2 outputs are compatible with nwb-schema 2.2.1
1. NWB2 outputs are compatible with the [NWB icephys extension proposal](https://docs.google.com/document/d/1cAgsXv26BmQoVfa7Greyxs0oc4IGH-t5aJsm-AwUAAE/)
    - this should be implied by backwards compatibility of this proposal
1. multiple input NWB2 file types supported (particularly AIBS vs HBG; covered by dataset creation dynamic dispatch)

non-goals
---------
1. NWB2 outputs make use of the NWB icephys extension proposal.
1. NWB1 files are supported
1. This module runs the pynwb validator (covered by writing using pynwb + roundtrip tests)
1. sweeps present will based on those in the input NWB2 file (this eliminates the old undesirable behavior in which failing sweeps were removed)

inputs
------
1. An NWB2 file
    - if sweep metadata is also argued, all sweeps for which metadata is argued must be present in the NWB file's sweep table
1. cell-level metadata. This is given as a sequence of named metadata fields along with an optional specifier for whether this data ought to be written to NWB or another sink (defaults based on icephys metadata list)
1. sweep-level metadata. This is a mapping from sweep identifiers to dictionaries of metadata
1. specification of output sinks. Must include at least an NWB2 file (with specified format), as well as all metadata-level requested sinks

outputs
-------
1. an nwb file with attached metadata
1. for each output sink, a file of metadata

cell-level fields
-----------------

These are effectively notes taken from the NWB2 DANDI metadata list, which seems to be a work in progress. We'll target these and adjust as necessary.

| name | destination (default) | lims field | notes |
| ---- | --------------------- | ---------- | ----- |
| Species | non-nwb | donors.organisms.name | |
| Subject ID | nwb (Subject.subject_id) | donor.id | |
| Subject age | non-nwb | donors.ages.days | format of this is unclear |
| Subject gender/sex | non-nwb | donors.gender.name | these seem like different fields |
| Subject date of birth | non-nwb | donors.date_of_birth | format again unclear |
| Specimen ID | nwb (???) | specimens.id | identifier for cell |
| Citation policy | nwb (???) | ??? | |
| Institution | nwb (NWBFile.institution) | AIBS | |
| Genotype | non-nwb | genotypes.full_genotype? | |
| Cre line | non-nwb | genotypes.? | how does this relate to genotype? |
| External solution recipe | nwb (???) | ??? | |
| Recording temperature | nwb | | |
| Reporter (e.g. Cre) status | nwb | genotypes.? | how does this relate to genotype? |
| Electrode ID | nwb (IntracellularElectrode.name?) | equipment.? | |
| Electrode resistance | nwb (IntracellularElectrode.resistance) | ??? | doc says str; prob typo |
| Electrode internal solution recipe | nwb (???) | ??? | |
| | | | |
| | | | |
| | | | |


sweep-level fields
------------------

| name | destination | description |
| ---- | ----------- | ----------- |
| stimulus_name | sweep table | |
| stimulus_interval | sweep table | |
| stimulus_amplitude | sweep table | |
| stimulus_type_name | sweep table | |
| stimulus_unit | sweep table | |

approach
--------

This is actually pretty straightforward (famous last words). Something like the following could work:

```Python

class MetadataSink(abc.ABC):
    """ Abstract base for metadata sinks.
    """

    @abstractproperty
    def supported_cell_level(self) -> List[str]:
        """ Report names of cell-level metadata which are supported
        """

    @abstractproperty
    def supported_sweep_level(self) -> List[str]:
        """ Report names of sweep-level metadata which are supported
        """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """ Must construct an internal store and set any parameters that might impact metadata registration (for instance, adding an AibsDataset so that sweep-level metadata is associated with existing sweeps)

        Parameters
        ----------
        *args, **kwargs : Implementors should specify appropriate parameters on a case-by-case basis

        """

    @abc.abstractmethod
    def register(self, name: str, value: Any, sweep_id: Optional[int] = None):
        """ Attaches a named piece of metadata to this sink's internal store. Should dispatch to a protected method which carries out appropriate validations and transformations.

        Parameters
        ----------
        name : the well-known name of the metadata
        value : the value of the metadata (before any required transformations)
        sweep_id : If provided, this will be interpreted as sweep-level metadata and sweep_id will be used to identify the sweep to which value ought to be attached. If None, this will be interpreted as cell-level metadata

        Raises
        ------
        ValueError : An argued piece of metadata is not supported by this sink

        """

    @abc.abstractmethod
    def serialize(self, *args, **kwargs):
        """ Write registered metadata to an external store

        Parameters
        ----------
        *args, **kwargs : Implementors should replace these with meaningful parameters. For instance, a JSON sink should require an output path, along with kwargs for json.dump

        """



class AibsNwb2MetadataSink(MetadataSink):

    @property
    def supported_cell_level(self) -> List[str]:
        """ Report names of cell-level metadata which are supported
        """

    @property
    def supported_sweep_level(self) -> List[str]:
        """ Report names of sweep-level metadata which are supported
        """

    def __init__(self, dataset: AibsDataset):
        """ Construct a metadata sink suitable for writing data to an AIBS nwb2 file
        """

        self.dataset = dataset.copy()

    def register(self, name: str, value: Any, sweep_id: Optional[int] = None):
        """
        """

    def serialize(self, to_nwb: Union[NWBHDF5IO, str, Path]):
        """ Writes this sink's dataset to an NWB2 file at the argued location
        """

        nwbfile = self.dataset.to_nwb2()
        # ... make an NWBHDF5IO and write

    @classmethod
    def from_nwb_path(path: Union[str, path]) -> "AibsNwb2MetadataSink":
        """ Utility for constructing an AibsNwb2MetadataSink from a path alone
        """


class DandiYamlMetadataSink(MetadataSink):

    @property
    def supported_cell_level(self) -> List[str]:
        """ Report names of cell-level metadata which are supported
        """

    @property
    def supported_sweep_level(self) -> List[str]:
        """ Report names of sweep-level metadata which are supported
        """

    def __init__(self):
        """ Construct a metadata sink suitable for writing data to a DANDI-compatible YAML
        """

        self.store = {}

    def register(self, name: str, value: Any, sweep_id: Optional[int] = None):
        """
        """

    def serialize(self, path: Union[str, Path]):
        """ Writes this sink's dataset to a yaml file at the argued location
        """


def attach_metadata(
    sinks: Dict[str, MetadataSink],
    cell_metadata: Dict[str, Any],
    sweep_metadata: Dict[int, Dict[str, Any]],
):
    """Attaches metadata inplace to a collection of sinks. Metadata can be provided at the cell and sweep levels.

    Parameters
    ----------
    sinks : Configured sinks for outputs.
    cell_metadata : Each has a name, a value, and a desired collection of sinks.
    sweep_metadata : Each has a name, a value, a sweep_id, and a desired collection of sinks.
    """

def run_attach_metadata(
    sink_configs: Sequence[Dict],
    cell_metadata: Dict[str, Any],
    sweep_metadata: Dict[int, Dict[str, Any]],
) -> Dict:
    """Attaches metadata to a set of sinks, serializes those sinks to external stores.

    Parameters
    ----------
    sink_configs : Each is used to construct a MetadataSink
    cell_metadata : Each has a name, a value, and a desired collection of sinks.
    sweep_metadata : Each has a name, a value, a sweep_id, and a desired collection of sinks.

    Returns
    -------
    lightweight information identifying external stores for sinks
    """

```

Of course, the specifics of e.g. `AibsNwb2MetadataSink` depend a lot on concurrent work in refactoring ipfx's NWB reading and writing. However, this architecture isolates that specificity to the internals of each sink.

We could make this a bit more rigorous by replacing some of the dicts (e.g. sink_configs, cell_metadata / sweep_metadata entries) with dataclasses or namedtuples at cost of more code. I think this would be a good idea.


Testing
-------

The most important test strategy is roundtripping; given some metadata, can we write it to sinks and then read it back? Alongside unit tests, a small harness for such roundtripping covers the great majority of our features.

A tricky requirement is icephys extension compatibility. Basically, we need to list this extension as a test requirement, then come up with a way to register it to pynwb for the duration of some tests (the same as the above - parametrization would be great) and then unregister the extension so that other tests are not impacted. On the other hand, that extension is supposed to be backwards compatible ...

Presentation
------------

The NWB team has been working with Kitware of NWB2 visualization, in particular for icephys. We should put together a notebook demonstrating this new functionality by viewing an NWB2 file before and after.

Review Notes
------------

t.b.d

Implementation Notes
--------------------

