Results
=======

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


