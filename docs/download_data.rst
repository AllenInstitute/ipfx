Data Access
===========

The electrophysiology data files for the PatchSeq experiments released by the
Allen Institute are stored in `Neurodata Without Borders 2.0 <https://nwb.org>`_ (NWB) format.
The files are hosted on the `Distributed Archives for Neurophysiology Data Integration (DANDI) <https://dandiarchive.org>`_.

The PatchSeq data release is composed of  two archives:

Mouse data archive (114 GB): `<https://dandiarchive.org/dandiset/000020>`_

Human data archive (12 GB): `<https://dandiarchive.org/dandiset/000023>`_

You can download these data following the above links or
using DANDI's command line client that can be installed as:

.. code-block:: bash

    pip install dandi

With the client installed, you can easily download individual files or an entire archive as:

.. code-block:: bash

    dandi download --output-dir <DIRECTORY> <URL>

where <DIRECTORY> is the existing directory on your file system
and <URL> is the url path of a file or archive.

The paths to individual data files withing the archive are listed in the file manifest
(need links) under the ``archive_uri`` column.
Each row of the manifest file contains file information as well as corresponding ``cell_specimen_id`` identifying cells.
The manifest includes information about several different data modalities (see the ``technique`` column)
recorded from each cell.
The intracellular electrophysiological recordings stored on DANDI are denoted as
``technique`` = intracellular_electrophysiology.

Each file manifest is accompanied by the experiment metadata file (need links)
that can be used to select data files satisfying desired experimental conditions.
