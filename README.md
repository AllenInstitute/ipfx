Welcome to Intrinsic Physiology Feature Extractor (ipfx)
========================================================

ipfx is a python package for computing intrinsic cell features from electrophysiology data.  This includes:

    * action potential detection (e.g. threshold time and voltage)
    * cell quality control (e.g. resting potential stability)
    * stimulus-specific cell features (e.g. input resistance)

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline. For more information on ipfx, please see the [documentation](ipfx.rtfd.io) (especially the [installation instructions](https://ipfx.readthedocs.io/en/latest/installation.html) and [code examples](https://ipfx.readthedocs.io/en/latest/auto_examples/index.html)).

## Quickstart:

To run:

```bash
 $ cd ipfx/ipfx/bin
 $ python pipeline_from_nwb.py input_nwb_file
```
User must specify the OUTPUT_DIR inside the pipeline_from_nwb.py

Input:
* input_nwb_file: a full path to the NWB file with cell ephys recordings

Output:

 * pipeline_input.json: input parameters
 * pipeline_output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures


## Contributing

Thank you for your interest in contributing to IPFX! We welcome contributions, from bug reports and feature requests, to code. Please see the [contribution guide](CONTRIBUTING.md) for more information.