Welcome to Intrinsic Physiology Feature Extractor (ipfx)
==========================================

ipfx is a python 2/3 package for computing intrinsic cell features from electrophysiology data.  This includes:

    * action potential detection (e.g. threshold time and voltage)
    * cell quality control (e.g. resting potential stability)
    * stimulus-specific cell features (e.g. input resistance)

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline.

For usage and installation instructions, see the [documentation](https://ipfx.readthedocs.io/en/latest//).

Contributing
------------
We welcome contributions! Please see our [contribution guide](CONTRIBUTING.md) for more information. Thank you!

Deprecation Warning
-------------------
The 1.0.0 release of ipfx brings some new features, like NWB2 support, along with improvements to our documentation and testing. We will also drop support for
- NWB1
- Python 2

Older versions of ipfx will continue to be available, but may receive only occasional bugfixes and patches.
