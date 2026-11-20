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

Asynchronous optimization
-------------------------

Import :class:`tpcp.aio.optimize.GridSearch` to use the task-based action.
Its ``optimize(dataset)`` method returns a bound task. Await it inside an
execution context or a worker to populate the search results.

.. autosummary::
   :toctree: generated/aio
   :template: class_with_private.rst

    tpcp.aio.optimize.BaseOptimize
    tpcp.aio.optimize.DummyOptimize
    tpcp.aio.optimize.Optimize
    tpcp.aio.optimize.GridSearch
    tpcp.aio.optimize.GridSearchCV

Optimize fits a synchronous pipeline in one CPU or GPU slot. A Worker stores
the fitted pipeline, transformed dataset and any optimization information in
the fit result. Each await applies that state to the optimizer you called.
The async optimizer also binds run and safe_run to predict with its fitted
pipeline. Await either method to obtain the pipeline with its prediction results.
GridSearchCV fits each fold and hyperparameter combination once when candidates
differ only in pure parameters. It scores candidate/fold pairs in the shared
pool and refits the best candidate on the full dataset when requested.

The optional tpcp.aio.optimize.optuna module exports CustomOptunaOptimize and
OptunaSearch. Each await runs a serial study in a CPU slot on a local thread,
preserving access to live storage and callbacks. Set n_jobs=1. Each call
invokes Optuna's study creation or loading policy and executes trials.
Configure persistent storage and load_if_exists to continue a named study.

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

.. autosummary::
   :toctree: generated/aio
   :template: class_with_private.rst

    tpcp.aio.optimize.optuna.CustomOptunaOptimize
    tpcp.aio.optimize.optuna.OptunaSearch
