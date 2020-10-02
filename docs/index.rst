.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   quick_start
   tutorial
   download_data
   stimuli
   auto_examples/index
   pipeline
   API Documentation <ipfx>
   Github <https://github.com/alleninstitute/ipfx>
   authors
   Releases <https://github.com/alleninstitute/ipfx/releases>


Welcome to Intrinsic Physiology Feature Extractor (IPFX)
--------------------------------------------------------

IPFX is a Python package for computing intrinsic cell features from electrophysiology data.
With this package you can:

    * Perform cell data quality control (e.g. resting potential stability)
    * Detect action potentials and their features (e.g. threshold time and voltage)
    * Calculate features of spike trains (e.g., adaptation index)
    * Calculate stimulus-specific cell features

For a full list of features refer to the WHOLE-CELL ELECTROPHYSIOLOGY FEATURE ANALYSIS Section in the `Electrophysiology Overview <https://help.brain-map.org/download/attachments/8323525/CellTypes_Ephys_Overview.pdf?version=2&modificationDate=1508180425883&api=v2>`_

You can use ``IPFX`` for tasks varying in scale from analyzing individual sweeps,
to datasets of single cells saved in the NWB file format,
to constructing a standalone data processing pipeline alike the
Allen Institute electrophysiology :doc:`pipeline` to process thousands of datasets

To get started check out the quick tutorial :doc:`tutorial`
or dive into complete examples :ref:`examples-index`.
