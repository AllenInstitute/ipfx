"""A shim for backwards compatible imports of EphysDataSet
"""

from allensdk.deprecated import class_deprecated

from ipfx.dataset.ephys_data_set import EphysDataSet
EphysDataSet = class_deprecated(  # type: ignore
    "Import EphysDataSet from ipfx.dataset.ephys_dataset rather than "
    "ipfx.ephys_dataset"
)(EphysDataSet)
