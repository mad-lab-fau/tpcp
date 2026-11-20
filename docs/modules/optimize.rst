tpcp.optimize: Optimization helper
==================================

.. automodule:: tpcp.optimize
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: tpcp.optimize

.. autosummary::
   :toctree: generated/optimize
   :template: class_with_private.rst

    Optimize
    GridSearch
    GridSearchCV
    DummyOptimize
    BaseOptimize

Asynchronous grid search
------------------------

Import :class:`tpcp.aio.optimize.GridSearch` to use the task-based action.
Its ``optimize(dataset)`` method returns a bound task. Await it inside an
execution context or a worker to populate the search results.

.. autosummary::
   :toctree: generated/aio
   :template: class_with_private.rst

    tpcp.aio.optimize.GridSearch

Optuna Interface
----------------

.. automodule:: tpcp.optimize.optuna
    :no-members:
    :no-inherited-members:

.. currentmodule:: tpcp.optimize

.. autosummary::
   :toctree: generated/optimize
   :template: class_with_private.rst

    optuna.CustomOptunaOptimize
    optuna.OptunaSearch