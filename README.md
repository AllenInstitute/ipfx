Welcome to Intrinsic Physiology Feature Extractor (ipfx)
========================================================

ipfx is a python package for computing intrinsic cell features from electrophysiology data.  This includes:

    * action potential detection (e.g. threshold time and voltage)
    * cell quality control (e.g. resting potential stability)
    * stimulus-specific cell features (e.g. input resistance)

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline.

## Quickstart:

To run::

    bash
    $ cd ipfx/ipfx/bin
    $ python pipeline_from_nwb.py input_nwb_file

User must specify the OUTPUT_DIR inside the pipeline_from_nwb.py

Input:
* input_nwb_file: a full path to the NWB file with cell ephys recordings

Output:

 * pipeline_input.json: input parameters
 * pipeline_output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures


Deprecation Warning
-------------------
We are working towards a 1.0.0 release of ipfx! This will bring some new features, like NWB2 support, along with improvements to our documentation and testing. We will also drop support for
- NWB1
- Python 2

Older versions of ipfx will continue to be available, but may receive only occasional bugfixes and patches.
