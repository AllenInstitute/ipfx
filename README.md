Welcome to Intrinsic Physiology Feature Extractor (IPFX)
==========================================

IPFX is a Python package for computing intrinsic cell features from electrophysiology data. With this package you can:

    * Perform cell data quality control (e.g. resting potential stability)
    * Detect action potentials and their features (e.g. threshold time and voltage)
    * Calculate features of spike trains (e.g., adaptation index)
    * Calculate stimulus-specific cell features

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline.

For usage and installation instructions, see the [documentation](https://ipfx.readthedocs.io/en/latest//).

Quick Start
------------
For runing the pipeline, please see the [quick_start](docs/quick_start.rst) 

Contributing
------------
We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md) for more information. Thank you!

Deprecation Warning
-------------------
The 1.0.0 release of ipfx brings some new features, like NWB2 support, along with improvements to our documentation and testing. We will also drop support for
- NWB1
- Python 2

Older versions of ipfx will continue to be available, but may receive only occasional bugfixes and patches.
