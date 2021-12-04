.. parameters_

Parameters
==========

Parameters define the behaviour of an algorithm or a pipeline.
They are knobs I can turn to make the analysis behave like I want to.
In `tpcp` all parameters are defined in the `__init__` method of a pipeline/algorithm.
This means we can modify them when creating a new instance:

.. code-block:: python

    >>> my_algo = MyAlgo(para_1="value1", para_2="value2")
    >>> my_algo.para_1
    value1

The initialization of objects in `tpcp` never has side effects (with the exceptions of mutable handling TODO: link).
This means all parameters will be added to the instance using the same name and without modification.

Potential validation of parameters is only performed when the algorithm is actually run.
This also means we can modify the parameters, until we perform the actual run.
In general this should be done using the `set_params` methode:

.. code-block:: python

    >>> my_algo = my_algo.set_params(para_1="value1_new")
    >>> my_algo.para_1
    value1_new

This also allows us to set nested parameters, if the nested objects support a `set_params` methode:

.. code-block:: python

    >>> my_algo = MyAlgo(para_1="value1", nested_algo_para=MyOtherAlgo(nested_para="nested_value"))
    >>> my_algo = my_algo.set_params(para_1="value1_new", nested_algo_para__nested_para="new_nested_value")
    >>> my_algo.para_1
    value1_new
    >>> my_algo.nested_algo_para.nested_para
    new_nested_value
