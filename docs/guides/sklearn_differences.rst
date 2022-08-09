Sklearn Differences
===================

The core of the tpcp API is inspired by the scikit-learn API.
However, sklearn (by nature) is focused on supervised machine learning algorithms and hence, their focus is on making
"training" as easy as possible.
As tpcp was developed originally to work with "traditional" (aka non-ML) algorithms, the focus of our API is running the
algorithms.

This means, while the API is similar to sklearn, there are some differences and gotchas, in particular when you are
using tpcp together with sklearn:

Results and main actions
------------------------
In sklearn the main "action" is the fit method.
It stores results of the training as attributes (with trailing underscores) on the estimator object.
A second `predict` method can then be used to predict the labels of new data and get them directly as a return value of
this method.

For tpcp, "training" algorithms is of secondary concern.
This means our main "action" is running the algorithm (the name of this method can vary from algorithm to algorithm).
This main action methods can store one or multiple results on the object (like the `fit` method in sklearn).

For algorithms, that require training, we expect them to implement a `self_optimize` method.
However, this method is not allowed to write new results attributes on the object, but should only modify/write
parameters that can be set via the constructor of the object.
For example, if I have an algorithm with 10 internal weights that can be optimized by the `self_optimize` method, in
tpcp I should provide a list of 10 default values in the constructor of the algorithm.
These values can then be modified by the `self_optimize` method and used by the actual action method.

TL;DR: In tpcp the "results" are the results of actual applying an algorithm to data, where in sklearn the "results" are
the results of training an algorithm on data.

Cloning
-------
Both, sklearn and tpcp provide a clone method that work very similar.
The sklearn version will check for each element, if the element has a `get_params` method and will create a new
instance of the class with the same parameters.
This means, most tpcp objects can be cloned correctly using the sklearn version.

However, the tpcp version will only clone objects via the `get_params`/`set_params` interface, when they are actually
tpcp objects.
This means, tpcp's clone version will create a deepcopy of sklearn classifiers.
In result, sklearn classifiers will maintain their fit results when cloned with `tpcp.clone`.

The reason behind that is, that we consider the results of the fit method to be part of the "parameters" of an
algorithms, which should survive the clone method (see more in the `self_optimize` documentation).


Parameter
---------
In tpcp we consider everything that can be optimized a parameter of the algorithm.
In contrast to sklearn, this conceptually also includes the internal weights of a ML classifier (or similar).
In tpcp, we therefore expect these parameters to be "stored" in the algorithm constructor and be preserved when we clone
the algorithm.
