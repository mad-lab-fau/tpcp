Custom Algorithms and Pipelines
===============================

Classes
-------

.. currentmodule:: tpcp

.. autosummary::
   :toctree: generated/algorithm
   :template: class_with_private.rst

    Algorithm
    Pipeline
    OptimizablePipeline


Functions
---------

.. currentmodule:: tpcp

.. autosummary::
   :toctree: generated/algorithm
   :template: function.rst

    make_action_safe
    make_optimize_safe
    get_param_names
    get_action_params
    get_results

The :doc:`typing helpers <types>` include type variables for algorithms and pipelines that retain
the concrete subclass in generic code.
