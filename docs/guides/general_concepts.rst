General Concepts
================

Parameters
----------

Parameters define the behaviour of an algorithm or a pipeline.
They are knobs you can turn to make the analysis behave like you want to.
In `tpcp` all parameters are defined in the `__init__` method of a pipeline/algorithm.
This means we can modify them when creating a new instance:

.. code-block:: python

    >>> my_algo = MyAlgo(para_1="value1", para_2="value2")
    >>> my_algo.para_1
    value1

The initialization of objects in `tpcp` never has side effects (with the exceptions of mutable handling TODO: link).
This means all parameters will be added to the instance using the same name and without modification.

Potential validation of parameters is only performed when the algorithm is actually run.
This also means we can modify the parameters, until we apply the algorithm to data.
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




Results
-------

The most important thing about an algorithms are the results it produces.
In tpcp we store results as class attributes on the respective algorithm or pipeline instance.
To differentiate these attributes from the algorithm parameters, all the names should have a "_" as a postfix by
convention.

.. warning:: This is one of the clear differences to sklearn!
             In sklearn the trained models are stored as attributes with "_", but not the final prediction results.

These attributes are only populated, when the actual algorithm is executed.
This happens when the "action"-method of the algorithm or pipeline is called.
For pipelines this method is simply called "run".
For specific algorithms it might have a different name that better reflects what the algorithm does
(e.g. detect, extract, transform, ...).

.. code-block:: python

    >>> my_algo = MyAlgo()
    >>> my_algo.detected_events_
    Traceback (most recent call last):
        ...
    AttributeError: 'MyAlgo' object has no attribute 'detected_events_'
    >>> my_algo = my_algo.detect(data)
    >>> my_algo.detected_events_
    ...

Because, results are just instance attributes, a pipeline or algorithms can have any number of results (and even results
that are calculated on demand using the `@property` decorator)

.. code-block:: python

    >>> my_algo.another_result_
    ...
    >>> my_algo.detected_events_in_another_format_
    ...

Cloning
-------

In general results are not considered persistent.
This means they are deleted when you run the pipeline again, or if you create a new version using clone.

.. code-block:: python

    >>> my_algo = MyAlgo()
    >>> my_algo = my_algo.detect(data)
    >>> my_algo.detected_events_
    ...
    >>> my_algo_clone = my_algo.clone()
    >>> my_algo_clone.detected_events_
    Traceback (most recent call last):
        ...
    AttributeError: 'MyAlgo' object has no attribute 'detected_events_'


