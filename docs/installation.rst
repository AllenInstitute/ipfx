.. highlight:: shell

============
Installation
============

The easiest way to install ``ipfx`` is via pip:

.. code-block:: bash

    pip install ipfx

We suggest installing ``ipfx`` into a managed Python environment, such as those provided by `anaconda <https://anaconda.org/anaconda/anaconda-project>`_ or `venv <https://docs.python.org/3/library/venv.html>`_. This avoids conflicts between different Python and package versions.

Installing for development
--------------------------

If you wish to make contributions to ``ipfx`` (thank you!), you must clone the `git repository <https://github.com/alleninstitute/ipfx>`_. You will need to install `git-lfs <https://git-lfs.github.com/>`_ before cloning (we store some test data in lfs). Once you have cloned ``ipfx``, simply pip install it into your environment:

.. code-block:: bash

    pip install -e path/to/your/ipfx/clone

(-e installs in editable mode, so that you can use your changes right off the bat).

See :doc:`contributing` for more information.