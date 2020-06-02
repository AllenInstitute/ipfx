Data Sets and Stimuli
=====================

:py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` is the starting point for analyzing data with ipfx. Instances of this class provide a standardized interface to the stimuli, data, and metadata for a single experiment. 
If you have an instance of :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet`, you can use it to:

    1. Build a "sweep table" for a data set (``pandas`` ``DataFrame`` with specific, required column names).
    2. Given a sweep number, return ``t``, ``v``, and ``i``.
    3. Obtain a :py:class:`~ipfx.sweep.SweepSet`, which provides accessors to the stimuli, data, and metadata for a logical grouping of sweeps

If you intend to use ipfx for data analysis, your first order of business should be to obtain an :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` instance. 
This is particularly straightforward if your data are stored in `Neurodata Without Borders 2.0 <https://nwb.org>`_ (NWB) format. 
NWB defines a standard format for storing neurophysiological data, making it easy for IPFX to read your data into an :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet`:

.. code-block:: python

    from ipfx.dataset.create import create_ephys_data_set

    ds = create_ephys_data_set(nwb_file="example.nwb")
    long_squares = ds.filtered_sweep_table(stimuli=ds.ontology.long_square_names) # more on this next!
    sweep_set = ds.sweep_set(long_squares.sweep_number)

Stimulus Naming
---------------

Higher-level analyses available in ``ipfx`` are critically-dependent on stimulus type and naming conventions.  In the example above,
:py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` was able to find all of the long square sweeps because the stimuli in the stimulus table have been named according
to the Allen Institute's stimulus naming protocol.

Unless overwritten, all :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` instances create a default :py:class:`~ipfx.stimulus.StimulusOntology` based on
`stimulus_ontology.json <http://github.com/AllenInstitute/ipfx/blob/master/allensdk/ipfx/stimulus_ontology.json>`_.  The job of this class is to provide mechanisms
for searching for stimuli that have been tagged with standardized names.  For example, "``C1SSCOARSE150112``" is a short square stimulus with the associated tags "``Core 1``" (part of the basic
protocol used for all data sets), "``Short Square``" (3ms square pulse), and "``Coarse``" (large jumps between amplitudes while searching for an action potential).

To run end-to-end analyses on an NWB file, ``ipfx`` uses the naming conventions defined in the :py:class:`~ipfx.stimulus.StimulusOntology` to identify sweeps
by stimulus type that can be used to compute stimulus-specific features (e.g. use all long squares to identify rheobase).  This enables the following example:


.. code-block:: python

    from ipfx.dataset.create import create_ephys_data_set
    from ipfx.data_set_features import extract_data_set_features
    from ipfx.utilities import prepare_sweep_info

    data_set = create_ephys_data_set(nwb_file=nwb_path)
    data_set.sweep_info = prepare_sweep_info(data_set)

    cell_features, sweep_features, cell_record, sweep_records = extract_data_set_features(data_set)

This code block does a large number of things:

    1. Compute spike times and spike features for all current-clamp sweeps
    2. Compute long square response features (e.g. input resistance, membrane time constant)
    3. Compute short square response features
    4. Compute ramp response features
