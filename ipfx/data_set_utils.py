"""A shim for backwards compatible imports of create_data_set
"""

from allensdk.deprecated import deprecated

from ipfx.dataset.create import create_ephys_data_set
create_data_set = deprecated(  # type: ignore
    "Instead of using ipfx.data_set_utils.create_data_set, use "
    "ipfx.dataset.create.create_ephys_data_set"
)(create_ephys_data_set)
