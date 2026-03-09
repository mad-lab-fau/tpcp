tpcp.parallel: Helper for global settings during parallel execution
===================================================================

For practical caveats around joblib-based multiprocessing, worker state, serialization, and import costs, see
:ref:`multiprocessing_caveats`.

.. automodule:: tpcp.parallel
    :no-members:
    :no-inherited-members:

Functions
---------

.. currentmodule:: tpcp.parallel

.. autosummary::
    :toctree: generated/parallel
    :template: function.rst

    delayed
    register_global_parallel_callback
    remove_global_parallel_callback
