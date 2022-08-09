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
Such checks can be performed by :class:`~tpcp.optimize.Optimize` class or the :class:`~tpcp.make_optimize_safe`
decorators based on the provided parameter annotations.
Have a look at the documentation there to understand which checks are performed.

To see these parameter annotations in action, check out this `example <optimize_pipelines>`_.

.. note:: One special case of parameter annotations is the `tpcp.PureParameter`.
          It can be used to annotate a parameter that does **not** influence the `self_optmize` method of a pipeline.
          I.e. it is only used and relevant for the action method.
          This can be useful information for parameter search methods like :class:`~tpcp.optimize.GridSearchCV`, as they
          don't need to rerun the optimization when only pure parameters are changed.
          For :class:`~tpcp.optimize.GridSearchCV` such an optimization can be enabled via the `pure_parameters`
          parameter.


External Optimization vs `self_optimize`
----------------------------------------
When implementing a new algorithm or pipeline that should have optimizable parameter, you need to decide whether to
implement an explicit `self_optimize` method or use (or create) an external parameter optimizer like the
:class:`~tpcp.optimize.GridSearch`.

The simple advise here is, that you should never "re-implement" any form of "dumb" search within a `self_optimize`
method.
The `self_optimize` should only be used, if there are algorithm specific details or methods that can be used to optimize
parameters far more efficient than random search (or similar).
For example, the backpropagation logic for a neuronal network would be a candidate for `self_optimize`.
It is domain specific (i.e. not generic) and hence, is less suited for a general "parameter" optimizer class.

However, at the end the line between to two domains is a bit fuzzy.
You might very well decide to implement something in the `self_optimize` method, and later decide to move this logic
into a more generic optimizer class.
Or you might start with a generic GridSearch and move to a `self_optimize` method, once you realise, you need very
specific modifications for your algorithm or group of algorithms.