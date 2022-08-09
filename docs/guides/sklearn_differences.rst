Sklearn Differences
===================

The core of the tpcp API is inspired by the scikit-learn API.
However, sklearn (by nature) is focused on supervised machine learning algorithms and hence, their focus is on making "training" as easy as possible.
As tpcp was developed originally to work with "traditional" (aka non-ML) algorithms, the focus of our API is running the algorithms.

This means, while the API is similar to sklearn, there are some differences and gotchas, in particular when you are using tpcp together with sklearn:

Cloning
-------
