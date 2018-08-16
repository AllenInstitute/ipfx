Welcome to allensdk.ipfx
========================

allensdk.ipfx is a python 2/3 package for computing intrinsic cell features from electrophysiology data.  This includes:

    * action potential detection (e.g. threshold time and voltage)
    * cell quality control (e.g. resting potential stability)
    * stimulus-specific cell features (e.g. input resistance)

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline.

##Quickstart:

To run:

```bash
 $ cd allensdk/ipfx/bin
 $ python run_pipeline_nwb.py input_nwb_file output_dir
```

Input:
 input_nwb_file: a full path to the NWB file with cell ephys recording
 output_dir: a base output directory which will include the output subdirectory

The output subdirectory is named from the basename of the nwb file.

Output:
 pipeline_input.json: input parameters
 pipeline_output.json: output including cell features
 output.nwb: NWB file including spike times
 log.txt: run log
 qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures

