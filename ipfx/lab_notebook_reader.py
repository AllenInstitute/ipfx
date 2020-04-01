"""A shim for backwards compatible imports of lab_notebook_reader
"""

from allensdk.deprecated import class_deprecated

from ipfx.dataset.labnotebook import LabNotebookReader
LabNotebookReader = class_deprecated(  # type: ignore
    "Import LabNotebookReader from ipfx.labnotebook rather than "
    "ipfx.lab_notebook_reader"
)(LabNotebookReader)
