Optimization and Training
=========================
.. _optimization:

In `tpcp`, we use the term *Optimization* as a wrapper term for any form of data-driven parameter optimization.
This can be traditional ML training of model weights, black-box optimizations of hyperparameters or a simple grid search
of thresholds in classical algorithms.
Therefore, we attempt to have a unified interface for all these cases.

This is achieved by defining "optimization" as any form of data-driven optimization of the "parameters"
(see :ref:`parameters`) specified in the `__init__` of an algorithm.
This optimization can be performed via *internal* optimization implemented in a `self_optimize` method on the pipeline
or via *external* optimization like the :class:`~tpcp.optimize.GridSearch` wrapper.

.. code-block:: python

    >>> from tpcp.optimize import GridSearch
    >>>
    >>> my_pipeline = MyPipeline(val1="initial_value")
    >>> gs = GridSearch(my_pipeline, {"val1": ["optimized_value_1", "optimized_value_2"]})
    >>> gs = gs.optimize(train_data)
    >>> my_optimized_pipeline = gs.optimized_pipeline_
    >>> my_optimized_pipeline.val1
    "optimized_value_2"

For pipelines that implement a `self_optimize` method, it is recommended to use the :class:`~tpcp.optimize.Optimize`
wrapper instead of calling `self_optimize` directly.

.. code-block:: python

    >>> from tpcp.optimize import Optimize
    >>>
    >>> my_optimizable_pipeline = MyOptimizablePipeline(val1="initial_value")
    >>> my_optimized_pipeline = Optimize(my_optimizable_pipeline).optimize(train_data).optimized_pipeline_
    >>> my_optimized_pipeline.val1
    "optimized_val1"

Parameter Annotations
---------------------

When talking about optimization it becomes clear, that we need to differentiate the different types of parameters an
algorithm or pipeline might have.
They can fall into three categories:

1. optimizable parameters: These parameters represent values and models that are/can be optimized using the
   `self_optimize` method.
2. hyper-parameters: These are parameters that change, how the optimization in `self_optimize` is performed.
3. "normal" parameters: Basically everything else. These parameters do neither influence nor are influenced by
   `self_optimize`. They only influence the output of the `action` method of the pipeline. See the
   `evaluation guide <algorithm_evaluation>`_ to better understand the distinction between parameters and
   hyper-parameters.

To make these distinction clear (for human and machine), `tpcp` provides a set of Type hints that can be applied
class level parameters to annotate the respective parameters:

.. code-block:: python

    >>> from tpcp import OptimizableParameter, HyperParameter, Parameter
    >>>
    >>> class MyOptimizablePipeline(OptimizablePipeline):
    ...     nn_weight_vector: OptimizableParameter[np.ndarray]
    ...     simple_threshold: Parameter[int]
    ...     my_hyper_parameter: HyperParameter[float]
    ...
    ...     def __init__(self, nn_weight_vector: np.ndarray, simple_threshold: int, my_hyper_parameter: float):
    ...         ...

This helps not only with documentation, but can actually be used to perform sanity checks when running the optimization.
For example, if after running `self_optimize` of a pipeline is called, none of the optimizable parameters is changed,
likely something has gone wrong.
This is done using the :class:`~tpcp.optimize.Optimize` class or the :class:`~tpcp.make_optimize_safe` decorators.
Have a look at the documentation there to understand which checks are performed.

To see these parameter annotations in action, check out this `example <optimize_pipelines>`_.
