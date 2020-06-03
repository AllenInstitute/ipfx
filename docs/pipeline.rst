QC and Analysis Pipeline
========================

The IPFX package is utilized in the Allen Institute's electrophysiology
processing pipeline that is being used to generate publicly available
data.

Given the nwb2 file with electrphysiological recording for a single cell the pipeline will perform
the following steps (run with the corresponding executables):

1. Compute QC features (``ipfx.bin.run_sweep_extraction``)
    Extract sweeps and their metadata from the nwb file
    Computed QC features for the cell as a whole and also for each individual sweeps

2. Perform QC checks (``ipfx.bin.run_qc``)
    Check QC criteria on QC features at the level of the entire cell and for individual sweeps
    Tag cell, sweeps failing QC criteria

3. Compute intrinsic features (``ipfx.bin.run_feature_extraction``)
    Compute features of spikes, spike trains, as well as stimulus specific features

4. Attach metadata (``ipfx.attach_metadata``)
    Add ancillary information about an experiment to the output nwb2 file

Each executable defines a schema for the input parameters specified
in the <input.json> and can be invoked as:

.. code-block:: bash

    python -m <executable> --input_json <path/to/input.json> --output_json <path/to/output.json>

where <executable> stands for the executables listed in the above pipeline steps.

Running the pipeline requires two additional pieces of information:

1. Stimulus ontology that maps names of sweeps in the input nwb2 file to the stimulus types known to ``ipfx``
2. QC criteria that specify the acceptable values for the computed QC features

If not explicitly provided, the pipeline will invoke the default values from the ipfx.defaults folder

