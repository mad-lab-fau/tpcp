The fundamental components: Datasets, Algorithms, and Pipelines
===============================================================

For a typical data analysis we want to apply an algorithm to a dataset.
This usually requires you to write: (a) some code to load and format your data, (b) your actual algorithm, and (c) some
sort of glueing code that brings both sides together.
To ensure reusability, it is a good idea to keep (c) explicitly separate from (a) and (b), aka you don't want your data
loading to be specific for your algorithm or your algorithm interface specific to your dataset.

To ensure that the interface of an algorithms should only require the input data it really needs and all inputs should
use as simple datastructures as possible.
For example, this means that an algorithm should only get the data of a single recording as input and not a
datastructure containing multiple recordings.
Looping over multiple recordings and/or participants should be something handled by the glueing code.

.. code-block:: python

    # Bad idea:
    def run_algorithm(dataset: CustomDatasetObject):
        ...

    # Better:
    def run_algorithm(imu_data: np.ndarray, sampling_rate_hz: float):
        ...

If multiple algorithms can be used equivalently (e.g. two algorithms to detect R-Peaks in an ECG signal), you should
ensure that the interfaces of the algorithms are identical or at least as similar as possible, so that your gluing code
requires minimal modification when changing algorithms.
To make this idea of a shared interface easier, we represent Algorithms as classes in tpcp that get all there algorithm
specific configuration via the init.

.. note::
    **Algorithms** are simple classes that get configuration parameters during initialisation and that have an "action"
    method, that can be used to apply the algorithm to some data.
    All algorithms should be subclasses of :class:`~tpcp.Algorithm`.
    If two algorithms can perform the same functionality, their action methods should adhere to the same interface.
    Some algorithm might further define a `self_optmize` method that is able to "train" certain input parameters based
    on provided data.

With your data loading code you usually want to abstract the complexity of data loading and provide a simple to use
interface to your data for your glueing code independent of the actual format and structure of the data on disc.
To make writing glueing code as simple as possible, it is a good idea to follow some form of standards with the loaded
data.
This could be standards that are designed for yourself, for your work group, or your entire scientific field.
The only important thing is that you are consistent, whenever you write data loading code.
As an example, you should always provide data in the same units after loading and represent it with the same (ideally
simple) data structure (e.g. 3D Acceleration is always a numpy array of shape 3xn with axis order x,y,z and all values
in m/s).
Using any form of standards, means that you can reuse a lot of your glueing code across multiple datasets.

Going one step further, in tpcp each dataset is a custom class inheriting from :class:`~tpcp.Dataset`.
This ensures that independent of the actual data you are working with (tabular, metadata, timeseries, some crazy
combination of everything), a common "standardized" datatype exists that can be used by high level utility functions
like :func:`~tpcp.validation.cross_validate`.

.. note::
   **Datasets** are custom classes that inherit from :class:`~tpcp.Dataset`.
   At their core each dataset class only provides an index of all the data that is available.
   This makes it possible for generic utility functions to iterate or split datasets.
   It is up to the user to add additional methods and properties to a dataset that represent the actual data that can
   be used by an algorithm.


In the ideal case this leads to a scenario, where you can use the same glueing code to run multiple different
algorithms on multiple different datasets, because they all share common interfaces.
In tpcp we call this glueing code Pipeline.

.. note::
    **Pipelines** are custom classes with a strictly defined interface that subclass :class:`~tpcp.Pipeline`.
    They have a single `run` method, that takes a instance of a :class:`~tpcp.Dataset` representing a single datapoint
    as input.
    Within the run method the pipeline is expected to retrieve the required data from the dataset object, pass it to one
    or multiple algorithms and provide results in a format that make sense for a given application.
    Some pipelines might additionally define a `self_optmize` method that is able to "train" certain input parameters
    based on provided data.

.. figure:: ../diagrams/algos_simple.svg

    In a simple case, a single pipeline can interface between all available Datasets and all Algorithms, because they
    share a common interface.

However, it is usually impossible to produce the exact same data interface for multiple different datasets, even within
the same domain.
Datasets might have different measurement procedures and different measurement modalities.
In the same way, you might have different types of analysis you want ot perform and hence, require the use of different
algorithms.
This means, you will often end up with multiple pipelines (even within a single project) that connect one data interface
(that might be shared by multiple datasets) with multiple algorithm interfaces for different types of analysis.

.. figure:: ../diagrams/algos_complicated.svg

   Pipelines act as glueing code for one Dataset interface with one or multiple Algorithm interfaces to perform one
   specific analysis.

Note, that even though we consider these as different pipelines, as they are designed for different analysis, they
might still share code (e.g. use the same utility functions, or have a common parent class), so that writing a new
Pipeline is often very easy.