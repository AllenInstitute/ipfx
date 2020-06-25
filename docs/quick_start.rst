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


Step-by-step guideline
----------------------

 * create a clean environment with conda

.. code-block:: bash

    conda create -n myenv python=3.6
    conda activate myenv

 * install ipfx from pypi with "pip install ipfx"

 * download example datasets (mouse data ~114 GB) with dandi

.. code-block:: bash

    pip install dandi
    dandi download https://dandiarchive.org/dandiset/000020

 * pick one dataset, e.g. sub-599387254_ses-601506492_icephys.nwb from mouse data, create output directory for saving the results, for example

.. code-block:: bash

     mkdir outputs
     mkdir outputs/qc

then run ipfx feature extraction

.. code-block:: bash

    python -m ipfx.bin.run_pipeline_from_nwb_file sub-599387254_ses-601506492_icephys.nwb outputs --qc_fig_dir outputs/qc

The extract features will be saved into outputs/sub-599387254_ses-601506492_icephys/output_json. The figures can be found in outputs/qc.
