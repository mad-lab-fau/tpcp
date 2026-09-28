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

Splitters and dataset labels
----------------------------

Native tpcp splitters return lists of group labels for each train and test fold. Use
``dataset.get_subset(group_labels=labels)`` to inspect a fold. Raw sklearn splitters still return
positional indices when called directly; ``DatasetSplitter`` adapts them to group labels.

``CombinedSplitter`` applies each child splitter to a selected part of the current dataset. The
children must have the same number of folds. For example, to cross-validate real recordings and
always include artificial recordings in training::

    from tpcp.validate import CombinedSplitter, DatasetSplitter, NoSplit

    cv = CombinedSplitter(
        (lambda ds: ds.get_subset(recording_type="real"), DatasetSplitter(5, groupby="participant")),
        (lambda ds: ds.get_subset(recording_type="artificial"), NoSplit(5, train=lambda ds: ds)),
    )

Selectors receive the dataset passed to the current splitter, so the same ``cv`` can be used on a
subset in nested validation. A selector's group labels must belong to its input dataset. The
labels determine the fold assignment, so selecting only some rows of a group still assigns that
whole group. A child may also be a raw sklearn splitter such as ``KFold(5)``.

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
