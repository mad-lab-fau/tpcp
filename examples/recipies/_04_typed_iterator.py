"""

TypedIterator
=============.

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
aggregations = [
    ("result_1", lambda _, results: sum(results)),
]

# %%
# 3. We create a new instance of `TypedIterator` with the result type and the aggregations.
from tpcp.misc import TypedIterator

iterator = TypedIterator(ResultType, aggregations=aggregations)

# %%
# Now we can iterate over our data and get a result object for each iteration, that we can then fill with the results.
for d, r in iterator.iterate(data):
    r.result_1 = d * 3
    r.result_2 = r.result_1 * 2
    r.result_3 = r.result_2 - 4

# %%
# The aggregated results are now available as attributes of the iterator.
print(iterator.result_1_)
print(iterator.result_2_)
print(iterator.result_3_)

# %%
# The raw results are available as a list of dataclass instances.
iterator.raw_results_

# %%
# While this version of the code required a couple more lines, it is much easier to understand and reason about.
# It clearly separates the configuration from the actual code and the core pipeline code is much cleaner.

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
# For the aggregations, we want to concatenate the r-peak positions.
# The aggregation function gets the list of inputs as the first argument and the list of results as the second
# argument.
# We can use this to create a combined dataframe with a proper index.
#
# We turn the `n_r_peaks` into a dictionary, to make it easier to map the results back to the inputs.

aggregations = [
    ("r_peak_positions", lambda datapoints, results: pd.concat(results, keys=[d.group_label for d in datapoints])),
    ("n_r_peaks", lambda datapoints, results: dict(zip([d.group_label for d in datapoints], results))),
]

# %%
# Now we can create the iterator and iterate over the dataset.
from pathlib import Path

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from examples.datasets.datasets_final_ecg import ECGExampleData

iterator = TypedIterator(QRSResultType, aggregations=aggregations)

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"

dataset = ECGExampleData(data_path)

for d, r in iterator.iterate(dataset):
    r.r_peak_positions = QRSDetector().detect(d.data["ecg"], sampling_rate_hz=d.sampling_rate_hz).r_peak_positions_
    r.n_r_peaks = len(r.r_peak_positions)

# %%
# Finally we can inspect the results stored on the iterator.
# Note, that `r_peak_positions_` is a single dataframe now and not a list of dataframes.
iterator.r_peak_positions_

# %%
# The `n_r_peaks_` is still a dictionary, as excpected.
iterator.n_r_peaks_

# %%
# The raw results are still available a list of dataclass instances.
iterator.raw_results_

# %%
# And the inputs are stored as well.
iterator.inputs_
