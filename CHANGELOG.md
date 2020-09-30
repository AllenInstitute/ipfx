# Change Log
All notable changes to this project will be documented in this file.

## Unreleased

### Added
- `@lru_cache(maxsize=None)` decorator to `EphysNWBData._get_series()` to improve performance by using caching.

### Changed
- `EphysNWBData.get_sweep_data()` now uses cached values from `EphysNWBData._get_series()`
- Made error checking for `EphysNWBData._get_series()` more robust.

## [1.0.0] = 2020-06-30

1.0.0 is the first public release of IPFX. As of this version:
- IPFX is now [on pypi](https://pypi.org/project/IPFX/)! You can install it by typing `pip install ipfx` into your terminal.
- IPFX now supports [NeurodataWithoutBorders](https://www.nwb.org) version 2 in place of version 1.
- IPFX supports Python 3 in place of Python 2.
- Numerous features, [documentation](https://ipfx.readthedocs.io/en/latest/) updates, and bugfixes have been incorporated into IPFX.
