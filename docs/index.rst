.. toctree::
   :maxdepth: 1
   :hidden:

   installation
   authors
   data_sets
   auto_examples/index
   pipeline
   API Documentation <ipfx>
   history


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

    from ipfx.ephys_extractor import SpikeExtractor

    ext = SpikeExtractor()
    spikes = ext.process(t, v, i)

The ``spikes`` object is a ``pandas`` ``DataFrame`` where rows are action potentials and columns are features.


Detect Spike Train Features
---------------------------

Given a spike train ``DataFrame``, you can then compute a number of other features, like adaptation and latency.

.. code-block:: python

    from ipfx.ephys_extractor import SpikeTrainFeatureExtractor

    ext = SpikeTrainFeatureExtractor()
    features = ext.process(t, v, i, spikes) # re-using spikes from above


Stimulus-specific Analysis
--------------------------

To analyze all of the sweeps with a particular stimulus type (say, long square pulses), you'll first need to create
a :py:class:`~ipfx.ephys_data_set.SweepSet` object.  This is an object that groups together the stimuli and responses of a group of sweeps.

.. code-block:: python

    from ipfx.ephys_data_set import Sweep, SweepSet

    sweep_set = SweepSet([ Sweep(t=t0, v=v0, i=i0),
                           Sweep(t=t1, v=v1, i=i1),
                           Sweep(t=t2, v=v2, i=i2) ])

Now that we have this object, we can hand it to one of the stimulus-specific analysis classes.  You first need
to configure a :py:class:`~ipfx.ephys_extractor.SpikeExtractor` and :py:class:`~ipfx.ephys_extractor.SpikeTrainExtractor`:

.. code-block:: python

    from ipfx.ephys_extractor import SpikeExtractor, SpikeTrainExtractor
    from ipfx.stimulus_protocol_analysis import LongSquareAnalysis

    start, end = 1.02, 2.02 # start/end of stimulus in seconds
    spx = SpikeExtractor(start=start, end=end)
    spfx = SpikeTrainFeatureExtractor(start=start, end=end)

    analysis = LongSquareAnalysis(spx, spfx)
    results = analysis.process(sweep_set)

At this point ``results`` contains whatever features/objects the analysis instance wants to return.

Analyze a Data Set
------------------

You can compute all features available in a data set with the :py:meth:`~ipfx.data_set_features.extract_data_set_features` function.  To
use this you need to create an :py:class:`~ipfx.ephys_data_set.EphysDataSet` instance, and your data set will need to follow
some standardized stimulus naming conventions and have some prerequisite sweep types.  This is described
in more detail in :doc:`data_sets`. Once you've done this, the following is possible:


.. code-block:: python

    from ipfx.mies_nwb.mies_data_set import MiesDataSet
    from ipfx.data_set_features import extract_data_set_features

    data_set = MiesDataSet(nwb_filename='example.nwb')
    cell_features, sweep_features, cell_record, sweep_records = extract_data_set_features(data_set)

This concise code block does a large number of things:

    1. Compute spike times and spike features for all current-clamp sweeps
    2. Compute long square response features (e.g. input resistance, membrane time constant)
    3. Compute short square response features
    4. Compute ramp response features

Take a look at :doc:`data_sets` to find out more.

Analysis Pipeline
-----------------

``ipfx`` also contains the scripts that deploy features of this library into the Allen Institute's production pipeline.
Each script/module in the pipeline defines a schema to validate inputs using `argschema <http://github.com/AllenInstitute/argschema>`_.
For more details, see :doc:`pipeline`.


