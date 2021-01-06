# Change Log
All notable changes to this project will be documented in this file.

## Unreleased

### Added

### Changed

## [1.0.2] = 2021-01-06

Changed:
- Add features_state information to the pipeline output json
- Improve performance of loading sweeps from NWB2 files by using LRU cache
- More robust error checking when loading time_series

Bug fixes:
- Fix segment length rounding error in the DAT file converter

## [1.0.0] = 2020-06-30

1.0.0 is the first public release of IPFX. As of this version:
- IPFX is now [on pypi](https://pypi.org/project/IPFX/)! You can install it by typing `pip install ipfx` into your terminal.
- IPFX now supports [NeurodataWithoutBorders](https://www.nwb.org) version 2 in place of version 1.
- IPFX supports Python 3 in place of Python 2.
- Numerous features, [documentation](https://ipfx.readthedocs.io/en/latest/) updates, and bugfixes have been incorporated into IPFX.
