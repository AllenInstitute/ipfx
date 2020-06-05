Stimulus Types
==============

Higher-level analyses available in ``IPFX`` are critically-dependent on stimulus type and naming conventions.
Unless overwritten, all :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` instances create a default :py:class:`~ipfx.stimulus.StimulusOntology` based on
`stimulus_ontology.json <https://github.com/AllenInstitute/ipfx/blob/master/ipfx/defaults/stimulus_ontology.json>`_.
The instance of this class provides mechanisms for searching for stimuli that have been tagged with standardized names.

For example, in the default ontology `stimulus_ontology.json <https://github.com/AllenInstitute/ipfx/blob/master/ipfx/defaults/stimulus_ontology.json>`_ we find one of the entries:

.. code-block:: JSON

    [
        [
            "core",
            "Core 1"
        ],
        [
            "resolution",
            "Coarse"
        ],
        [
            "name",
            "Short Square"
        ],
        [
            "code",
            "C1SSCOARSE",
            "C1SSCOARSE150112"
        ]
    ],

where:
code: "``C1SSCOARSE150112``" and ""``C1SSCOARSE``"" are a stimulus codes found in the dataset
core: "``Core 1``" (part of the basic protocol used for all data sets),
name: "``Short Square``" (3ms square pulse),
resolution: "``Coarse``" (large jumps between amplitudes while searching for an action potential).

Tagging provides a mechanism for mapping stimulus codes to stimulus types and enables filtering of the sweeps based on stimulus types.

For example, Short Square stimuli are identified by the following name tags:

.. code-block:: python

        self.short_square_names = ( "Short Square",
                                    "Short Square Threshold",
                                    "Short Square - Hold -60mV",
                                    "Short Square - Hold -70mV",
                                    "Short Square - Hold -80mV" )

that allows mapping the sweep with the stimulus code "``C1SSCOARSE150112``" to
the Short Square stimuli 'self.short_square_names'.

With the ontology defined, you can now filter :py:class:`~ipfx.dataset.ephys_data_set.EphysDataSet` sweeps by the stimulus type:

.. code-block:: python

    short_square_table = data_set.filtered_sweep_table(
        stimuli=data_set.ontology.long_square_names
    )

that returns a table of metadata for the sweeps matching the ``self.short_square_names`` tags.
