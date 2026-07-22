The dataset base class
======================

.. automodule:: tpcp._dataset
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: tpcp

.. autosummary::
   :toctree: generated/dataset
   :template: class.rst

    Dataset
    DatasetWrapperMixin


Wrapping datasets
-----------------

:class:`~tpcp.DatasetWrapperMixin` contains the common implementation for datasets that wrap another dataset while
preserving a shared domain-specific dataset interface. It expands the wrapped index, derives the wrapper's initial
grouping, and resolves the wrapped datapoint represented by each wrapper datapoint.

Because the mixin does not inherit from :class:`~tpcp.Dataset`, wrapper classes combine it with their domain-specific
dataset base::

    class AugmentedImageDataset(
        DatasetWrapperMixin[ImageDataset],
        ImageDataset,
    ):
        ...

.. important::
   The inheritance order is required. ``DatasetWrapperMixin`` must be the first base and the domain-specific dataset
   must be the second base. Reversing them changes Python's method resolution order, preventing the mixin's
   ``create_index`` implementation from taking precedence. TPCP raises a :class:`TypeError` when it detects the reversed
   order.
