Tutorial
===========

This guide will walk you through the functionality of IPFX starting with the simplest examples
and build on itself towards increasing complexity of the scope of the analysis and the size of data to be analyzed.

Detect Action Potentials
------------------------

To detect action potentials, you must provide three 1D numpy arrays: stimulus current ``i`` in pA,
response voltage ``v`` in mV, and timestamps ``t`` in seconds.  With these three arrays, you can then do the following:

.. code-block:: python

    from ipfx.feature_extractor import SpikeFeatureExtractor

    ext = SpikeFeatureExtractor()
    spikes = ext.process(t, v, i)

The ``spikes`` object is a ``pandas`` ``DataFrame`` where rows are action potentials and columns are features.


Extract Spike Train Features
----------------------------

Given a spike train ``DataFrame``, you can then compute a number of other features(e.g. adaptation and latency:

.. code-block:: python

    from ipfx.feature_extractor import SpikeTrainFeatureExtractor

    ext = SpikeTrainFeatureExtractor()
    features = ext.process(t, v, i, spikes) # re-using spikes from above


Stimulus-specific Analysis
--------------------------

To analyze all of the sweeps with a particular stimulus type (say, long square pulses), you'll first need to create
a :py:class:`~ipfx.sweep.SweepSet` object. This object provides utilities for accessing properties of a group
of :py:class:`~ipfx.sweep.Sweep` objects:

.. code-block:: python

    from ipfx.sweep import Sweep, SweepSet

    sweep_set = SweepSet([ Sweep(t=t0, v=v0, i=i0),
                           Sweep(t=t1, v=v1, i=i1),
                           Sweep(t=t2, v=v2, i=i2) ])


Sweeps corresponding to the same stimulus type may have different stimulus start time.
This happens because sweeps may have different duration of the test pulse epoch preceding the stimulus epoch.
To perform the analysis, we must align a sweep set to have the same stimulus start times.

Here, we will align sweeps to the beginning of the experimental epoch,
that includes stimulus epoch padded by the pre-stimulus and post-stimulus time intervals.
With sweeps aligned, we can obtain common to all sweeps ``start_time`` and ``end_time`` time of the stimulus:

.. code-block:: python

    sweep_set.align_to_start_of_epoch("experiment")

    sweep = sweep_set[0]
    t = sweep.t
    start_idx, end_idx = sweep.epochs["stim"] # choose stimulus epoch
    start_time, end_time = t[start_idx], t[end_idx]

Now that we have this object, we can hand it to one of the stimulus-specific analysis classes.  You first need
to configure a :py:class:`~ipfx.feature_extractor.SpikeFeatureExtractor` and :py:class:`~ipfx.feature_extractor.SpikeTrainFeatureExtractor`:

.. code-block:: python

    from ipfx.feature_extractor import SpikeFeatureExtractor, SpikeTrainFeatureExtractor
    from ipfx.stimulus_protocol_analysis import LongSquareAnalysis

    spx = SpikeFeatureExtractor(start=start_time, end=end_time)
    spfx = SpikeTrainFeatureExtractor(start=start_time, end=end_time)

    analysis = LongSquareAnalysis(spx, spfx)
    results = analysis.process(sweep_set)

At this point ``results`` contains whatever features/objects the analysis instance wants to return.

Analyze a Data Set
------------------

The :py:meth:`~ipfx.data_set_features.extract_data_set_features` function allows you to calculate
all available features for a given dataset in one call.
IPFX supports datasets stored in `Neurodata Without Borders 2.0 <https://nwb.org>`_ (NWB) format
via a :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` class, which provides a well-known interface to all of the data in an experiment.
The data released by the Allen Institute is hosted on the DANDI public archive in the NWB format.
Refer to :doc:`download_data` page for the instructions on downloading the data files.

To create an instance of the :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet`:

.. code-block:: python

    from ipfx.dataset.create import create_ephys_data_set

    dataset = create_ephys_data_set(nwb_file="path/to/experiment.nwb")
    long_squares = dataset.filtered_sweep_table(stimuli=ds.ontology.long_square_names) # more on this next!
    sweep_set = dataset.sweep_set(long_squares.sweep_number)

where ``path/to/experiment.nwb`` is a local path to the nwb2 file that you have downloaded from the public archive.

With an instance of the :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` available you can easily obtain:
a :py:class:`~ipfx.sweep.Sweep` for a given sweep number:

.. code-block:: python

    sweep = ds.sweep(sweep_number)
    t = sweep.t
    v = sweep.v
    i = sweep.i

with the corresponding ``t``, ``v``, and ``i`` arrays.

You may also obtain a :py:class:`~ipfx.sweep.SweepSet`, for a particular grouping of sweeps
by filtering the ``sweep_table``:

.. code-block:: python

    long_squares = dataset.filtered_sweep_table(stimuli=dataset.ontology.long_square_names) # more on this next!
    sweep_set = dataset.sweep_set(long_squares.sweep_number)

where ``dataset.ontology`` includes references to the names of all stimuli types known to ``IPFX``.
See :doc:`stimuli` for details.

Finally, you can run end-to-end analyses on an NWB file:

to calculate the QC features:

.. code-block:: python

    from ipfx.qc_feature_extractor import sweep_qc_features, cell_qc_features

    dataset = create_ephys_data_set(nwb_file="path/to/experiment.nwb")
    sweep_qc_features = sweep_qc_features(dataset)
    cell_features, cell_tags = cell_qc_features(dataset)

and to calculate the analysis features:

.. code-block:: python

    from ipfx.data_set_features import extract_data_set_features

    drop_failed_sweeps(data_set)     # sweeps with incomplete recording or failing QC criteria
    (cell_features, sweep_features,
     cell_record, sweep_records,
     cell_state, feature_states) = dsft.extract_data_set_features(data_set)

this code block does the following:
    1. Creates a dataset
    2. Drops failed sweeps that cannot be used for feature extraction
    3. Computes features for the Long Square, Short Square and Ramp sweeps
