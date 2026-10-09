tpcp.validate: Validation helper
================================

.. automodule:: tpcp.validate
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: class_with_private.rst

    BaseDatasetSplitter
    CombinedSplitter
    DatasetSplitter
    NoSplit
    SubsetSplitter

Scoring
-------
.. currentmodule:: tpcp.validate


.. autosummary::
   :toctree: generated/validate
   :template: class_with_private.rst

    Scorer
    Aggregator
    FloatAggregator
    MacroFloatAggregator

.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: function.rst

    mean_agg
    no_agg


Functions
---------
.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: function.rst

    cross_validate
    validate

The :doc:`scoring aliases <types>` describe score functions, aggregator return values, and the
``scoring`` argument accepted by validation functions.
