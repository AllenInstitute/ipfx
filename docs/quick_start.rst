Quick Start
===========

To run:

.. code-block:: bash

    $ cd ipfx/ipfx/bin
    $ python pipeline_from_nwb.py input_nwb_file outputdir

Input:
* input_nwb_file: a full path to the NWB file with cell ephys recordings

Output:

 * pipeline_input.json: input parameters
 * pipeline_output.json: output including cell features
 * output.nwb: NWB file including spike times
 * log.txt: run log
 * qc_figs: index.html includes cell figures and feature table and sweep.html includes sweep figures
