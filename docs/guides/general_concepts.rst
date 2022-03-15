General Concepts
================

Parameters
----------
.. _parameters:

Parameters define the behavior of an algorithm or a pipeline.
They are knobs you can turn to make the analysis behave like you want to.
In `tpcp` all parameters are defined in the `__init__` method of a pipeline/algorithm.
This means we can modify them when creating a new instance:

.. code-block:: python

    >>> my_algo = MyAlgo(para_1="value1", para_2="value2")
    >>> my_algo.para_1
    value1

The initialization of objects in `tpcp` never has side effects (with the exceptions of `mutable handling <mutable_defaults>`_).
This means all parameters will be added to the instance using the same name and without modification.

Potential validation of parameters is only performed when the algorithm is actually run.
This also means we can modify the parameters until we apply the algorithm to data.
In general this should be done using the `set_params` method:

.. code-block:: python

    >>> my_algo = my_algo.set_params(para_1="value1_new")
    >>> my_algo.para_1
    value1_new

This also allows us to set nested parameters if the nested objects support a `set_params` method:

.. code-block:: python

    >>> my_algo = MyAlgo(para_1="value1", nested_algo_para=MyOtherAlgo(nested_para="nested_value"))
    >>> my_algo = my_algo.set_params(para_1="value1_new", nested_algo_para__nested_para="nested_value_new")
    >>> my_algo.para_1
    value1_new
    >>> my_algo.nested_algo_para.nested_para
    nested_value_new

It is important to understand that in `tpcp` everything can/is a parameter.
This includes simple threshold parameters or entire sklearn/pytorch models.
The latter becomes important, when we are talking about optimizing algorithms.
Because only parameters exposed in the init can be optimized in tpcp.
This means, if you want to use tpcp to train a neuronal network, some data structure representing the network must be
one of the parameters.
You can learn more about that in the `optimization guide <optimization>`_.

Results
-------

The most important thing about an algorithm are the results it produces.
In `tpcp` we store results as class attributes on the respective algorithm or pipeline instance.
This allows algorithms to provide multiple outputs, without having complex return types of your action methods.
To differentiate these attributes from the algorithm parameters, all the names should have a "_" as a suffix by
convention.

.. warning:: This is one of the clear differences to `sklearn`!
             In sklearn the trained models are stored as attributes with "_", but not the final prediction results.

These attributes are only populated when the actual algorithm is executed.
This happens when the "action" method of the algorithm or pipeline is called.
For pipelines this method is simply called `run`.
For specific algorithms it might have a different name that better reflects what the algorithm does
(e.g., "detect", "extract", "transform", ...).

.. code-block:: python

    >>> my_algo = MyAlgo()
    >>> my_algo.detected_events_
    Traceback (most recent call last):
        ...
    AttributeError: 'MyAlgo' object has no attribute 'detected_events_'
    >>> my_algo = my_algo.detect(data)
    >>> my_algo.detected_events_
    ...

Because results are just instance attributes, a pipeline or algorithms can have *any* number of results (and even results
that are calculated on demand using the `@property` decorator)

.. code-block:: python

    >>> my_algo.another_result_
    ...
    >>> my_algo.detected_events_in_another_format_
    ...

Because results are stored on an algorithm instance, calling the action method (`detect` in this example) again, will
overwrite the results.
This means, if you need to generate results for e.g. multiple data points, you need to store the values of the result
attributes in a different data structure or create a new algorithm instance before you apply the action again.
The latter can be done using **cloning**.

Cloning
-------

In `tpcp` it is often required to create a copy of an algorithm or pipeline with *identical configuration*.
For example, when iterating over a dataset and applying an algorithm to each data point, you want to have a "fresh"
instance of the algorithm to eliminate any chance of train-test leaks and to not overwrite the results stored on the
algorithm object.
In `tpcp` we use the `clone` method for that.
It creates a new instance of an algorithm with the same parameters. All parameters are copied as well in case they are nested algorithms or other complex structures.

.. code-block:: python

    >>> my_algo = MyAlgo(para=3)
    >>> my_algo.para
    3
    >>> my_algo_clone = my_algo.clone()
    >>> my_algo_clone.para
    3
    >>> my_algo_clone.set_params(para=4)
    >>> my_algo_clone.para
    4
    >>> my_algo.para
    3

Results and other modifications to an algorithm or pipeline instance are not considered persistent.
This means they are deleted when cloning the pipeline.

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

For more complex situations, it is important to understand how we handle nested parameters in a little more detail.
When cloning an algorithm or pipeline, we also attempt to clone each parameter.
If the parameter is an instance of :class:`~tpcp.BaseTpcpObject` or any subclass, we clone it in the same way as the
main algorithm.
This means that for these objects only their parameters will be copied over to the new object.
For all other objects, we will use :func:`copy.deepcopy`, which will create a full memory copy of the object.
This ensures that the clone is fully independent of the original object.

If a parameter is a list or a tuple of other values, we will iterate through them and clone each value individually.

.. warning::
    Getting a deepcopy of parameters that are not algorithms is usually what you would expect, but might be surprising,
    when one of your parameters is a `sklearn` classifier.
    When using the `sklearn` version of :func:`~sklearn.base.clone`, it will strip the trained model attributes from the
    object.
    The `tpcp` version will keep them.
    The reason for that is that in `tpcp`, we consider the trained model a *parameter* and **not** a *result*.
    Hence, we need to copy it over to the new algorithm instance.


Mutable Defaults
----------------
.. _mutable_defaults:

.. warning::
    Whenever you use `list`, `dicts`, `np.arrays`, `pd.Dataframe` or other `mutable <https://www.geeksforgeeks.org/mutable-vs-immutable-objects-in-python>`_ container types,
    instances of `tpcp` objects, `sklearn` classifiers, or any kind of other custom class instance as default values to a
    class parameters, wrap them in :func:`~tpcp.cf`!

    To understand why, keep reading.

Mutable defaults are a bit of an
`unfortunate trap in the Python language <https://florimond.dev/en/posts/2018/08/python-mutable-defaults-are-the-source-of-all-evil>`_.
Simply put, if you use a mutable object like a `list`, a `dict`, or an instance of a custom class as default value to any
parameter of a class, this object will be shared with **all instances** of that class:

.. code-block:: python

    >>> class MyAlgo:
    ...     def __init__(self, para=[]):
    ...         self.para = para
    >>>
    >>> first_instance = MyAlgo()
    >>> first_instance.para.append(3)
    >>> second_instance = MyAlgo()
    >>> second_instance.para.append(4)
    >>> second_instance.para
    [3, 4]
    >>> first_instance.para
    [3, 4]


These types of issues are usually hard to spot and, in the case of nested algorithms, might even lead to train-test leaks!

The usual workaround would be to set the default value to `None` or to some other value that indicates "no value provided" and
then later replace it with the actual default value.

However, this is something you might easily forget and, usually, makes the whole thing harder to read, as you might need
to dig through multiple layers of function calls and inheritance to find the actual default value.
Since we expect you to write a lot of custom classes when working with `tpcp`, this means these workarounds might become cumbersome, and the chance you are using mutable defaults by accident can be
quite high (talking from experience).

In `tpcp` we use two measures against that.
First, we have a very basic detection for mutable objects in the `__init__` signature and raise an `Exception` if we detect
one.
Note that we only explicitly check for a couple of common mutable types! Thus, you should still keep mutable defaults in mind, in
particular when you are working with non-standard objects and class instances as `__init__` parameters.

We apply this check to all objects that inherit from our base classes.
This means the class above would have raised an error at creation time:

.. code-block:: python

    >>> from tpcp import Algorithm
    >>> class MyAlgo(Algorithm):
    ...     def __init__(self, para=[]):
    ...         self.para = para
    Traceback (most recent call last):
        ...
    tpcp.exceptions.MutableDefaultsError: The class MyAlgo contains mutable objects as default values (['para']). ...

Second, we have a simple workaround called the :class:`~tpcp.CloneFactory` or the short alias :func:`~tpcp.cf`.
Wrapping the mutable with this factory will create a clone of the object for every new instance.
Of course this only works for classes that inherit from our base classes!

.. code-block:: python

    >>> from tpcp import Algorithm, cf
    >>> class MyAlgo(Algorithm):
    ...     def __init__(self, para=cf([])):
    ...         self.para = para
    >>> first_instance = MyAlgo()
    >>> first_instance.para.append(3)
    >>> second_instance = MyAlgo()
    >>> second_instance.para.append(4)
    >>> second_instance.para
    [4]
    >>> first_instance.para
    [3]
