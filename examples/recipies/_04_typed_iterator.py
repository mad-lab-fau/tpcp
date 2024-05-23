"""
.. _typed_iterator:

TypedIterator
=============

This example shows how to use the :class:`~tpcp.misc.TypedIterator` class, which might be helpful, when iterating over
data and needing to store multiple results for each iteration.

The Problem
-----------
A very common pattern when working with any type of data is to iterate over it and then apply a series of operations
to it.
In simple cases you might only want to store the final result, but often you are also interested in intermediate or
alternative outputs.

What typically happens, is that you create multiple empty lists or dictionaries (one for each result) and then append
the results to them during the iteration.
At the end you might apply further operations to the results, e.g. aggregations.

Below is a simple example of this pattern:
"""

data = [1, 2, 3, 4, 5]

result_1 = []
result_2 = []
result_3 = []

for d in data:
    intermediate_result_1 = d * 3
    result_1.append(intermediate_result_1)
    intermediate_result_2 = intermediate_result_1 * 2
    result_2.append(intermediate_result_2)
    final_result_3 = intermediate_result_2 - 4
    result_3.append(final_result_3)

# An example aggregation
result_1 = sum(result_1)

print(result_1)
print(result_2)
print(result_3)

# %%
# Fundamentally, this pattern works well.
# However, it does not really fit into the idea of declarative code that we are trying to achieve with tpcp.
# While programming, there are 3 places where you need to think about the result and the result types.
# This makes it harder to reason about the code and also makes it harder to change the code later on.
# In addition, the main pipeline code, which should be the most important part of the code, is cluttered with
# boilerplate code concerned with just storing the results.
#
# While we could fix some of these issues by refactoring a little, with `TypedIterator` we provide (in our opinion)
# a much cleaner solution.
#
# The basic idea of `TypedIterator` is to provide a way to specify all configuration (i.e. what results to expect and
# how to aggregate them) in one place at the beginning.
# It further simplifies how to store results, by inverting the data structure.
# Instead of worrying about one data structure for each result, you only need to worry about one data structure for each
# iteration.
# Using dataclasses, these objects are also typed, preventing typos and providing IDE support.
#
# Let's rewrite the above example using `TypedIterator`:
#
# 1. We define our result-datatype as a dataclass.
from dataclasses import dataclass


@dataclass
class ResultType:
    result_1: int
    result_2: int
    result_3: int


# %%
# 2. We define the aggregations we want to apply to the results.
#    If we don't want to aggregate a result, we simply don't add it to the list.
#    We provide some more explanation on aggregations below, just accept this for now.
from tpcp.misc import TypedIteratorResultTuple


def sum_agg(results: list[TypedIteratorResultTuple[int, ResultType]]):
    return sum(r.result.result_1 for r in results)


aggregations = [
    ("result_1", sum_agg),
]

# %%
# 3. We create a new instance of `TypedIterator` with the result type and the aggregations.
# We use the "square bracket" typing syntax to bind the output datatype and the input datatype we are planning to
# iterate over.
# This way, our IDE is able to autocomplete the attributes of the result type.
from tpcp.misc import TypedIterator

iterator = TypedIterator[int, ResultType](ResultType, aggregations=aggregations)

# %%
# Now we can iterate over our data and get a result object for each iteration, that we can then fill with the results.
for d, r in iterator.iterate(data):
    r.result_1 = d * 3
    r.result_2 = r.result_1 * 2
    r.result_3 = r.result_2 - 4

# %%
# You can access the data using the ``results_`` attribute.
iterator.results_

# %%
# Your IDE should be able to autocomplete the attributes.
iterator.results_.result_1

# %%
# The raw results are available as a list of Result tuples.
# They allow us to access the results in the order they were created, and contain further metadata like the input data.
iterator.raw_results_

# %%
# While this version of the code required a couple more lines, it is much easier to understand and reason about.
# It clearly separates the configuration from the actual code and the core pipeline code is much cleaner.
#
# A real-world example
# --------------------
# Below we apply this pattern to a pipeline that iterates over an actual dataset.
# The return types are a little bit more complex to show some more advanced features of aggregations.
#
# For this example we apply the QRS detection algorithm to the ECG dataset demonstrated in some of the other examples.
# The QRS detection algorithm only has a single output.
# Hence, we use the "number of r-peaks" as a second result here to demonstrate the use case.
#
# Again we start by defining the result dataclass.
import pandas as pd


@dataclass
class QRSResultType:
    """The result type of the QRS detection algorithm."""

    r_peak_positions: pd.Series
    n_r_peaks: int


# %%
# Our input data is going to be a dataset object of the ECGExampleData type.
from pathlib import Path

from examples.datasets.datasets_final_ecg import ECGExampleData

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"

dataset = ECGExampleData(data_path)

# %%
# For the aggregations, we want to concatenate the r-peak positions.
# The aggregation function gets all raw results as input.
# So it can access all inputs, all results, and all metadata.
# This means you can define any aggregation you want.
# In this case, we want to concatenate the r-peak positions into a single dataframe.
# And we turn the `n_r_peaks` into a dictionary, to make it easier to map the results back to the inputs.
#
# Note that we can type these functions using the `TypedIteratorResultTuple` type.
# Like the iterator itself, this type is generic and allows you to specify the input and output types.
# So in our case, the input is `ECGExampleData` and the output is `QRSResultType`.
from typing_extensions import TypeAlias

from tpcp.misc import TypedIteratorResultTuple

result_tup: TypeAlias = TypedIteratorResultTuple[ECGExampleData, QRSResultType]


def concat_r_peak_positions(results: list[result_tup]):
    return pd.concat({r.input.group_label: r.result.r_peak_positions for r in results})


def aggregate_n_r_peaks(results: list[result_tup]):
    return {r.input.group_label: r.result.n_r_peaks for r in results}


aggregations = [
    ("r_peak_positions", concat_r_peak_positions),
    ("n_r_peaks", aggregate_n_r_peaks),
]

# %%
# Now we can create the iterator and iterate over the dataset.
# The iterator takes the same type parameters as our result-tuple.
#
# We can then iterate over the dataset and apply the QRS detection algorithm.
from examples.algorithms.algorithms_qrs_detection_final import QRSDetector

qrs_iterator = TypedIterator[ECGExampleData, QRSResultType](QRSResultType, aggregations=aggregations)

for d, r in qrs_iterator.iterate(dataset):
    r.r_peak_positions = QRSDetector().detect(d.data["ecg"], sampling_rate_hz=d.sampling_rate_hz).r_peak_positions_
    r.n_r_peaks = len(r.r_peak_positions)

# %%
# Finally we can inspect the results stored on the iterator.
qrs_iterator.results_

# %%
# Note, that `r_peak_positions_` is a single dataframe now and not a list of dataframes.
qrs_iterator.results_.r_peak_positions

# %%
# The `n_r_peaks_` is still a dictionary, as expected.
qrs_iterator.results_.n_r_peaks

# %%
# The raw results are still available.
qrs_iterator.raw_results_


# %%
# Custom Iterators
# ----------------
# When passing an iterable directly is not really convenient, you can also create a custom iterator class.
# This class can reimplement ``iterate`` with custom logic.
# For example, you could provide a custom iterator that takes a data and a sections parameter and then loops over the
# sections of the data.
#
# For this we need to create a custom subclass inheriting from ``BaseTypedIterator``.
from collections.abc import Iterator
from typing import Generic, TypeVar

from tpcp.misc import BaseTypedIterator

CustomTypeT = TypeVar("CustomTypeT")


class SectionIterator(BaseTypedIterator[pd.DataFrame, CustomTypeT], Generic[CustomTypeT]):
    def iterate(self, data: pd.DataFrame, sections: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, CustomTypeT]]:
        # We turn the sections into a generator of dataframes
        data_iterable = (data.iloc[s.start : s.end] for s in sections.itertuples(index=False))
        # We use the `_iterate` method to do the heavy lifting
        yield from self._iterate(data_iterable)


# %%
# We create some dummy data and sections to test the iterator.
dummy_data = pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
dummy_sections = pd.DataFrame({"start": [0, 5], "end": [5, 10]})


# %%
# Now we can use the iterator to iterate over the data.
# We skip any form of aggregation here, as it is not really relevant for this example, but it would work the same way
# as before.
@dataclass
class SimpleResultType:
    n_samples: int


custom_iterator = SectionIterator[SimpleResultType](SimpleResultType)

for d, r in custom_iterator.iterate(dummy_data, dummy_sections):
    print(d)
    r.n_samples = len(d)

# %%
# We can see that the iterator iterated over the two sections of the data.
# And the raw results contain two instances of the result dataclass.
custom_iterator.raw_results_

# %%
custom_iterator.results_

# %%
custom_iterator.results_.n_samples

# %%
# Advanced Usacases
# -----------------
# For a really advanced use cases, check out `mobgap GsIterator
# <https://mobgap.readthedocs.io/en/latest/modules/generated/pipeline/mobgap.pipeline.GsIterator.html>`_.
# This makes use of sub-iterations to allow to iterate and aggregate subregions of the data dynamically.
#
# Additional Aggregators
# ++++++++++++++++++++++
# We allow to pass additional aggregators to the iterator that have names that are not part of the result type.
# This allows to perform additional aggregations.
# They work as before, but the aggregation results are not available on the result object, but rather as raw dictionary
# via the ``additional_results_`` attribute.
# We show that below with the section iterator we defined above.
aggregations = [("sum_n_samples", lambda results: sum(r.result.n_samples for r in results))]

custom_iterator = SectionIterator[SimpleResultType](SimpleResultType, aggregations=aggregations)

for d, r in custom_iterator.iterate(dummy_data, dummy_sections):
    r.n_samples = len(d)

custom_iterator.results_
# %%
custom_iterator.additional_results_
