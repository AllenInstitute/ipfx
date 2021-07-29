# Change Log
All notable changes to this project will be documented in this file.

## Unreleased

### Added

### Changed
- Round the duration calculation in `run_feature_vector_extraction` so that
vectors of the same length are produced even when floating point approximations
of times are different.

## [1.0.4] = 2021-07-29
Changed:
- selects recording rather than sweep epoch by default
- restructures epoch detection to use recording epoch for stim epoch detection
- Use NaNs instead of truncating

Bug fixes:
- Round the duration calculation in `run_feature_vector_extraction` so that
vectors of the same length are produced even when floating point approximations
of times are different.

## [1.0.3] = 2021-02-02
Changed:
- Adds new 'Stimulus contains NaN values' tag and error handling to qc_feature_extractor

Bug fixes:
- Fixed memory leak in method of `EphysNWBData` caused by `@lru_cache` decorator

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
