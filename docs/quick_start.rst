Quick Start
===========

To get started, install IPFX via pip into a fresh conda evironment. For details see the Installation guide :doc:`installation`.

Next, example datasets can be downloaded according to the instructions in the Data Access :doc:`download_data`.

To process a dataset saved in the nwb file, run:

.. code-block:: bash

    $ python -m ipfx.bin.run_pipeline_from_nwb_file <input_nwb_file> <output_dir>

Input:
 
 * input_nwb_file: a full path to the NWB file with cell ephys recordings
 * output_dir: an output directory to save the results
 

Output:

 * input.json: input parameters
 * output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures

