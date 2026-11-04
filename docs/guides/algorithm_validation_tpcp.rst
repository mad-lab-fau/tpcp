Algorithm Validation in tpcp
============================

.. note:: If you are unsure about algorithm validation from a general scientific point of view, have a look at our
          general guide :ref:`here <algorithm_evaluation>` first.


Pre-Requisites
--------------
To use the algorithm validation tools in tpcp, you need to first represent your data as a :class:`~tpcp.Dataset` and
implement the algorithms you want to validate as :class:`~tpcp.Pipeline`.
All parameters that should be optimized (either internally or using an external wrapper) as part of a parameter search
should be exposed as parameters in the init.

Train-Test Splits
-----------------
As part of any validation for algorithms that require any form of data-driven optimization you need to perform create a
hold-out test set.
For this purpose you can simply use the respective functions from sklearn
(:func:`sklearn.model_selection.train_test_split`).
In case you are planning to use crossvalidation (next section), you can also use any of the sklearn CV splitter
(e.g. :class:`sklearn.model_selection.KFold`).

As :class:`~tpcp.Dataset` classes implement an iterator interface the train-test splits work just like with any other
list like structure.
Have a look at the :ref:`custom_dataset_basics` example for practical examples.

The only important thing you need to keep in mind is, that in tpcp we put all information into a single object.
This means we don't have a separation between data and labels on a data-structure level.
In case you need to perform a stratified or a grouped split, you need to temporarily create an array with the required
labels (for stratified split) or groups (for grouped split) and then pass it as `y` to the `split` method of your
splitter.

For a stratified split, this might look like this:

>>> from sklearn.model_selection import StratifiedKFold
>>>
>>> splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
>>> data = CustomDatasetClass(...)
>>> label_array = [d.label for d in data]
>>> for train_index, test_index in splitter.split(data, label_array):
...     train_data = data[train_index]
...     test_data = data[test_index]
...     # do something with the data


For a grouped split it might look like this:

>>> from sklearn.model_selection import GroupKFold
>>>
>>> splitter = GroupKFold(n_splits=2)
>>> data = CustomDatasetClass(...)
>>> # You can use `create_string_group_labels` method to create an array of group labels based on the dataset index
>>> groups = data.create_string_group_labels("patient_groups")
>>> for train_index, test_index in splitter.split(data, groups=groups):
...     train_data = data[train_index]
...     test_data = data[test_index]
...     # do something with the data

This works well, when you iterate over your folds on your own.
If you are planning to use :func:`~tpcp.validate.cross_validate` you need to handle these special cases a little
different.
More about that in the next section.

Cross Validation
----------------
Instead of doing a single train-test split, a cross-validation is usually preferred.
Analog to the sklearn function we provide a :func:`~tpcp.validate.cross_validate` function.
The api of this function is as similar as possible to the sklearn function.

Have a look at the full example for cross-validate for basic usage: :ref:`cross_validation`.

A couple of things you should keep in mind:

- The first parameter must be an **Optimizer**, not just an optimizable Pipeline.
  If you have an optimizable pipeline you want to cross-validate withour external parameter search, you need to wrap it
  into an :class:`~tpcp.optimize.Optimize` object.
- If you want to use a pipeline without Optimization in the cross-validate function, you can wrap it in an
  :class:`~tpcp.optimize.DummyOptimize` object.
  This object has the correct optimization interface, but does not perform any optimization.
  In such a case you would usually not need to use a cross-validation, but it might be helpful to run a non-optimizable
  algorithm on the exact same folds than an optimizable algorithm you want to compare it to.
  This way you get comparable means and standard deviations over the cross-validation folds
- If you want to use stratified or grouped splits, you need to create the arrays for the labels or groups as above and
  then pass it as the `groups` or `mock_labels` parameter.
  Note that the `mock_labels` will really only be used for the CV splitter and not for the actual evaluation of the
  algorithm.

If you use multiprocessing via `n_jobs`, also review :ref:`multiprocessing_caveats`.

.. _per_side_subset_selection:

Select subsets after fold assignment
------------------------------------
Use :class:`~tpcp.validate.DatasetSplitter` to assign related recordings together
and optionally select a different subset on each side. Its ``train_selector`` and
``test_selector`` callbacks receive only the Dataset assigned to their respective
side of each fold, after the base splitter applies grouping and stratification.
Each callback returns a Dataset with group labels from that side. Returning labels
outside the assigned side raises ``ValueError`` to prevent cross-fold leakage.
``None`` keeps that side unchanged. Selected labels follow the returned Dataset's
order; without a selector, the base splitter's order is preserved.

Keep one datapoint per recording and pass ``groupby="participant"`` or
``groupby="device"`` to the splitter to keep related recordings together.
This differs from ``Dataset.groupby``, which merges index rows into datapoints.
If one grouped datapoint contains several test types or sources, selecting some
rows still returns its group label. Consumers select the whole original datapoint
by that label, so this cannot represent selecting half of a grouped datapoint.

For example, retain all tests of training participants and evaluate only their
held-out counterparts' continuous walking tests. Leave ``train_selector`` unset
and set only ``test_selector``. Training then includes its participants' continuous
walking tests too; every recording of the held-out participants stays out of
training. See :ref:`participant_test_selection` for an executable example.

These selectors define the experimental protocol. They do not augment data or
replace an optimizer's ``train_dataset_transform``. Pipelines and training code
can remain independent of recording sources and test types. Selectors run once
per requested fold on each configured side; ``get_n_splits`` does not call them.

:class:`~tpcp.validate.SubsetSplitter` instead selects before splitting.
:class:`~tpcp.validate.CombinedSplitter` unions its children's train contributions;
adding a train-only :class:`~tpcp.validate.NoSplit` child cannot remove recordings
already assigned to training by another child. ``NoSplit`` repeats a fixed
assignment and does not rotate participant folds.

Custom Scoring
--------------
In tpcp we assume that your problem is likely complex enough to require a custom scoring function.
Therefore, we don't provide anything pre-defined.
However, we want to make it as easy as possible to pass-through all the information you need to evaluate your algorithm.

A scoring function can return any number of metrics (as dict of values).
Even further we allow to return any non-numeric values (e.g. meta-data or "raw-results") from scoring functions
(a regular frustration I had with sklearn).
These non-numeric values can either be passed through all cross-validation or optimization methods by wrapping them
with :class:`~tpcp.validate.NoAgg` or passed through any form of custom aggregator (learn more about that
:ref:`here<custom_scorer>`).
