# Change Log
All notable changes to this project will be documented in this file.

## Unreleased

### Added

### Changed

## [2.0.0] = 2024-10-23
Changed:
- Removed Python 3.6 support
- Updated dependencies and library for Python 3.9 to 3.11 support
- Moved CI and testing to GitHub Actions

## [1.0.8] = 2023-06-29
Changed:
- Fixed an error that was obscuring underlying errors when trying to _get_series information for a PatchClampSeries

## [1.0.7] = 2022-12-5
Changed:
- Added StimulusType and STIMULUS_TYPE_NAME_MAPPING to stimulus ontology, replacing definitions in EphysDataset
- Updated data_set_features to use correct sweep feature extractor detection parameters based on StimulusType

## [1.0.6] = 2022-6-29
Changed:
- Stop IPFX from caching its NWB Schemas when writing/modifying NWB files

## [1.0.5] = 2021-12-13
Bug fixes:
- Converts nwb_version attribute to string if it is in utf-8 encoded bytes.

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
