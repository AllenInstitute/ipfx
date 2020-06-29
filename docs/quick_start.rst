Quick Start
===========

To process a dataset saved in the nwb file, run:

.. code-block:: bash

    $ python -m ipfx.bin.run_pipeline_from_nwb_file <input_nwb_file> <output_dir> --qc_fig_dir <qc_fig_dir>

Input:
 
 * input_nwb_file: a full path to the NWB file with cell ephys recordings
 * output_dir: an output directory to save the results
 * qc_fig_dir: an output directory to save qc figures
 

Output:

 * input.json: input parameters
 * output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures
