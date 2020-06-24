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

 * pipeline_input.json: input parameters
 * pipeline_output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures


Electrophysilogy Data
---------------------
The Distributed Archives for Neurophysiology Data Integration (DANDI) hosts electrophysiology files. DANDI supports data download using HTTP and the DANDI command line client [pip install dandi].

The paths to individual data files are listed in the [file manifest]. Directory paths are available here:

 * Mouse data (114 GB): https://dandiarchive.org/dandiset/000020
 * Human data (12 GB): https://dandiarchive.org/dandiset/000023

Example:

Download all mouse ephys data:

.. code-block:: bash

    pip install dandi
    dandi download https://dandiarchive.org/dandiset/000020
