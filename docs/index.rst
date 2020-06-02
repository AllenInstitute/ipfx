.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   authors
   data_sets
   auto_examples/index
   pipeline
   API Documentation <ipfx>
   Github <https://github.com/alleninstitute/ipfx>
   Releases <https://github.com/alleninstitute/ipfx/releases>


Welcome to Intrinsic Physiology Feature Extractor (ipfx)
--------------------------------------------------------

ipfx is a Python package for computing intrinsic cell features from electrophysiology data.  This includes:

    * action potential detection (e.g. threshold time and voltage)
    * cell quality control (e.g. resting potential stability)
    * stimulus-specific cell features (e.g. input resistance)

This software is designed for use in the Allen Institute for Brain Science electrophysiology data processing pipeline.


Want to cut to the chase?  Take a look at our :ref:`examples-index`.  Otherwise, continue on for a high-level overview.


Detect Action Potentials
------------------------

To detect action potentials, first construct three 1D numpy arrays: stimulus current in pA, response voltage in mV, and timestamps in seconds.  With these three arrays, you can then do the following:

.. code-block:: python

    from ipfx.feature_extractor import SpikeFeatureExtractor

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i)

The ``spikes`` object is a ``pandas`` ``DataFrame`` where rows are action potentials and columns are features.


Detect Spike Train Features
---------------------------

Given a spike train ``DataFrame``, you can then compute a number of other features, like adaptation and latency.

.. code-block:: python

    from ipfx.feature_extractor import SpikeTrainFeatureExtractor

    ext = SpikeTrainFeatureExtractor()
    features = ext.process(t, v, i, spikes) # re-using spikes from above


Stimulus-specific Analysis
--------------------------

To analyze all of the sweeps with a particular stimulus type (say, long square pulses), you'll first need to create
a :py:class:`~ipfx.sweep.SweepSet` object.  This is an object that groups together the stimuli and responses of a group of sweeps.

.. code-block:: python

    from ipfx.sweep import Sweep, SweepSet

    sweep_set = SweepSet([ Sweep(t=t0, v=v0, i=i0),
                           Sweep(t=t1, v=v1, i=i1),
                           Sweep(t=t2, v=v2, i=i2) ])

Now that we have this object, we can hand it to one of the stimulus-specific analysis classes.  You first need
to configure a :py:class:`~ipfx.feature_extractor.SpikeFeatureExtractor` and :py:class:`~ipfx.feature_extractor.SpikeTrainFeatureExtractor`:

.. code-block:: python

    from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
    from ipfx.stimulus_protocol_analysis import LongSquareAnalysis

    start, end = 1.02, 2.02 # start/end of stimulus in seconds
    spx = SpikeFeatureExtractor(start=start, end=end)
    spfx = SpikeTrainFeatureExtractor(start=start, end=end)

    analysis = LongSquareAnalysis(spx, spfx)
    results = analysis.process(sweep_set)

At this point ``results`` contains whatever features/objects the analysis instance wants to return.

Analyze a Data Set
------------------

The :py:meth:`~ipfx.data_set_features.extract_data_set_features` function allows you to calculate all available features for a given dataset in one call. 
This powerful functionality relies on the :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` class, which provides a well-known interface to all of the data in an experiment. 
For more information on construction and using :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet`, see :doc:`data_sets`.


Analysis Pipeline
-----------------

At the Allen Institute, we use ``ipfx`` in two ways:

1. as library code imported directly into scripts and notebooks. You can see examples of this use case in the inline code samples here and in the :ref:`examples-index`.
2. in an automated data processing pipeline

To support the latter use case ``ipfx`` includes a number of scripts for running sweep extraction, quality control, spike extraction, and feature extraction. To find out more, see :doc:`pipeline`