Data Access
===========

The electrophysiology data files for the PatchSeq experiments released by the
Allen Institute are stored in `Neurodata Without Borders 2.0 <https://nwb.org>`_ (NWB) format.
The files are hosted on the `Distributed Archives for Neurophysiology Data Integration (DANDI) <https://dandiarchive.org>`_.

The PatchSeq data release is composed of  two archives:

Mouse data archive (114 GB): `<https://dandiarchive.org/dandiset/000020>`_

Human data archive (12 GB): `<https://dandiarchive.org/dandiset/000023>`_

Each archive is accompanied by the corresponding file manifest and the experiment metadata tables.

The file manifest table contains information about the files included in the archive,
their location ("archive_uri" column) and the corresponding cell("cell_specimen_id" column).
The file manifest combines information about several different data modalities (see the "technique" column)
recorded from each cell. The files with the intracellular electrophysiological recordings stored on DANDI are denoted as
"technique" = intracellular_electrophysiology.

In turn, the experiment metadata table includes information about the experimental conditions
for each cell("specimen_id" column). This table could be used to select the desired cells
satisfying particular experimental conditions. Then, given the desired "specimen_ids",
you can find the corresponding DANDI urls of these data from the file manifest.

IPFX includes a utility that provides file manifest and experiment data of the published archives.

For example, to obtain detailed information about Human data archive:

.. code-block:: python

    from ipfx.data_access import get_archive_info
    archive_url, file_manifest, experiment_metadata = get_archive_info(dataset="human")

where ``archive_uri`` is the DANDI URL for the Human data,
``file_manifest`` is a pandas.DataFrame of file manifest and
``experiment_metadata`` is a pandas.DataFrame of experiment metadata.
To obtain the same information for the Mouse data, change to `dataset="mouse"` in the function argument.

You can download data files by directly entering the DANDI's archive_uri in your browser.
Alternatively, a more powerful option is to install DANDI's command line client:

.. code-block:: bash

    pip install dandi

With client installed, you can easily download individual files or an entire archive as:

.. code-block:: bash

    dandi download --output-dir <DIRECTORY> <URL>

where <DIRECTORY> is the existing directory on your file system
and <URL> is the url of a file or an archive.

