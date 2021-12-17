Optimization and Training
=========================

In tpcp, we use the term *Optimization* as a wrapper term for any form of data driven parameter optimization.
This can be traditional ML training of model weights, black-box optimizations of hyper-parameters or a simple Gridsearch
of thresholds in classical algorithms.
Therefore, we attempt to have a unified interface for all these cases.

This is achieved, by defining "optimization" as any form of datadriven optimization of the "parameters"
(see :ref:`parameters`) specified in the `__init__` of an algorithm.
This optimization can be performed via *internal* optimization implemented in a `self_optimize` method on the pipeline
or via external optimization like the :ref:`~tpcp.optimization.GridSearch` wrapper.

.. code-block:: python

    >>> from tpcp.optimize import GridSearch
    >>>
    >>> my_pipeline = MyPipeline(val1="initial_value")
    >>> gs = GridSearch(my_pipeline, {"val1": ["optimized_value_1", "optimized_value_2"]})
    >>> gs = gs.optimize(train_data)
    >>> my_optimized_pipeline = gs.optimized_pipeline_
    >>> my_optimized_pipeline.val1
    "optimized_value_2"

For pipelines that implement a `self_optimize` method, it is recommended to use the :ref:`~tpcp.optimization.Optimize`
wrapper instead of calling `self_optimize` directly.

.. code-block:: python

    >>> from tpcp.optimize import Optimize
    >>>
    >>> my_optimizable_pipeline = MyOptimizablePipeline(val1="initial_value")
    >>> my_optimized_pipeline = Optimize(my_optimizable_pipeline).optimize(train_data).optimized_pipeline_
    >>> my_optimized_pipeline.val1
    "optimized_val1"
