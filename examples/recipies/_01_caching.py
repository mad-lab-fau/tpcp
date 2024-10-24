"""
.. _caching:

Caching
=======

With the way tpcp Datasets and Algorithms are structured, we encourage users to build out their entire pipeline as one
chain of function calls that can easily be rerun end-to-end, if parameters are changed.
In such an approach, storing intermediate results, or trained models, is discouraged.
This makes it easier to fully reproduce results and to be sure that you did not accidentally forget to update any
intermediate outputs.
However, the big downside of this approach is that it can be very slow, if you have a large dataset and/or a complex
calculations that have to be run over and over again, even-though the input data and parameters for that specific
processing step have not changed.

Caching provides the middle-ground.
It stores intermediate results in a transparent way (i.e. as a user of the code, you shouldn't even notice), tracks
dependencies (i.e. inputs) of cached function calls to automatically invalidate the cache, if the inputs change and
(usually) provides one central place to invalidate all cached results, making it easy to do a fully "clean" run.

This example shows a couple of ways of how caching can be integrated into :class:`~tpcp.Dataset` and
:class:`~tpcp.Algorithm`/ :class:`~tpcp.Pipeline` objects.

Before we start, some general information you need to be aware of:

1. All cache methods we show here, can only track the direct inputs of the function call.
   This means if your function relies on some global state, this will not be tracked and the cache will not be correctly
   invalidated, if this global state changes.
2. Caching class methods directly is discouraged, as they get the entire class instance as input.
   As instance parameters might change often and independent of the actual required inputs of the function, this can
   lead to a lot of unnecessary cache invalidations/cache misses.
   The only exception to this are our custom class decorators, which are designed to work with tpcp action methods
   specifically.
3. You should only ever cache functions that are deterministic, i.e. always return the same output for the same input.
4. You should only ever cache pure functions, i.e. functions that do not have any side-effects (modify global state,
   write to files, etc.).
5. Don't rely on correct cache invalidation, if you need highly reproducible results.
   For example, if you want to make the final run to create the final results of your paper, always manually delete the
   cache before running (if you where using a disk cache).

With that out of the way, let's get started.

In general, we separate two ways of caching:

1. Disk Caching: Disk caches store input and outputs of a function persistently on disk.
   This way, the cache can be reused between different runs of the same code or by multiple processes, when using
   multiprocessing.
   This can be slow, if inputs or outputs are large, and you are using a slow storage medium (e.g. a network drive).
   Hence, this is only useful for really expensive computations and not micro-optimizations.
2. Memory (aka RAM) Caching: Memory caches store input and outputs of a function in RAM.
   This is usually much faster than disk caching, but the cache is not persistent and will be lost, if the process is
   terminated.
   This means, this cache is only useful, if the same computation result is accessed multiple times within the same
   process/script.
   Also, your RAM space is usually much more limited than your disk space, so you need to be careful to not cache too
   much data.

Disk Caching
------------

The easiest way to perform disk-caching in Python is to use the `joblib <https://joblib.readthedocs.io/en/latest/>`__
library.
We highly recommend to read through their documentation, as it provides a general useful information on that topic.
Below, are just the most important points:

1. Joblib uses Pickle to serialize the inputs and outputs of a function.
   The pickle-output of the input data is hashed to check if the cache is still valid.
   This means, that you can only cache inputs that are pickle-able.
2. Pickling the inputs and storing the outputs on disk can be slow, if the data is large.
   In particular large input data will slow down the process, as pickling has to be performed every time the function is
   called and hence, can diminish the speedup you get from caching.
3. Joblib also stores the function body in the cache, so that the cache can be invalidated, if the function changes.
   However, joblib can not store dependencies of the function.
   This means, if your function calls other functions, they will not be stored in the cache and the cache will not be
   invalidated, if they change.
   This means, when you are still actively developing your code, you should disable caching to avoid running into
   problems with outdated caches.
   In general, it is a good idea to delete your cache from time-to-time when you are updating your code.

All caching in joblib is done via the :class:`~joblib.Memory` decorator.
This decorator takes a `location` argument, which defines where the cache should be stored.

Within custom objects
~~~~~~~~~~~~~~~~~~~~~
In many cases, in particular in Datasets, you that you are developing yourself, there is a specific substep of the
processing that is slow and you want to cache.
For example, loading a large file from disk and performing some pre-processing on it.
For efficient cashing, you want to wrap this slow function into the :class:`~joblib.Memory` decorator and then call
the cached version of the function every time you need the data.
To make the location parameter of the Memory object accessible to the user of your tpcp object, we recommend to add
a `memory` parameter to your `init` method, that can take a joblib memory instance.

Below, you can see an example Dataset that uses this pattern.
Note, that we factored out the processing that we want to cache into a global pure function and then cache this function
every time we call the data attribute.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Memory
from tpcp import Dataset


def _get_data(participant_id: int):
    # Here we would do an expensive load a pre-processing step that takes a long time.
    # For this example, we just return a constant value that depends on the participant_id.

    # This print statement will only be executed when we don't cache the function.
    print(f"Un-cached call of `_get_data` with participant_id {participant_id}")

    return np.arange(10) * participant_id


class DiskCachedDataset(Dataset):
    # Memory(None) is equivalent to no caching
    def __init__(
        self,
        memory: Memory = Memory(None),
        *,
        groupby_cols=None,
        subset_index=None,
    ) -> None:
        self.memory = memory
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _cached_get_data(self, participant_id: int):
        # Putting the cached function into a method is not strictly necessary, but makes it easier when you need to
        # use the cached function in multiple places in your class.
        # It also allows you to specify the call signature explicitly, which gives you autocomplete in your IDE.
        # Usually applying the cache decorator, will "confuse" your IDE and autocomplete will not work.
        return self.memory.cache(_get_data)(participant_id)

    @property
    def data(self):
        self.assert_is_single(None, "get_data")
        p_id = self.group_label.participant_id
        return self._cached_get_data(p_id)

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"participant_id": [1, 2, 3]})


# %%
# Now we can use the class without caching (default):

dataset = DiskCachedDataset()
dataset

# %%
# Because we don't cache the function, we see the print statement every time we access the data attribute:
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

# %%
# When we use caching, we only see the print statement once, indicating a cache hit.
#
# Note, that we clean the cache before and after in this example, to make sure that we don't get a cache hit from a
# previous run.
# Usually, you would not do this, as you want to reuse the cache between runs.
try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()

cache = Memory(HERE / ".cache")
cache.clear()
dataset = DiskCachedDataset(memory=cache)

dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data
cache.clear()

# %%
# If you want to clear to cache, you can use the `clear` method of the memory instance or just delete the cache folder.
#
# This pattern of caching is extensively used in the
# `gaitmap_datasets <https://github.com/mad-lab-fau/gaitmap-datasets>`__ package.
# Head over their to see more complex examples of this in action.
#
# Final Notes:
#
# 1. Dataset classes with slow to load file types and comples pre-processing are usually the best candidates for
#    caching.
#    These pieces of code rarely change, and you are not calling these load functions with various different parameters.
#    They are usually only called once per recording.
# 2. When caching large data, your disk space can quickly fill up.
#    To avoid this, avoid caching multiple steps of your pre-processing/loading individually and validate if the
#    performance gain is worth the disk space.
#
# Within existing objects or full action cashing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The method above could be considered "precision caching".
# We only cache the part of the processing that is slow and leave the rest of the processing as is.
# This is more flexible and usually the preferred approach.
#
# However, this assumes that we have full control over the code and can easily factor out the slow part.
# It also assumes that we have control over the instance creation of the object.
# In particular if you have an algorithm that is used multiple times in your pipeline, or even deeply nested within
# other algorithms, integrating caching can be difficult.
# Also, you might not want to explicitly implement caching for all algorithms in existence, just on the off chance that
# someone might need it.
#
# One alternative, is to replace a class method globally with a cached version.
# As mentioned above, this can be notoriously difficult, as you need to deal with the ever-changing mutable nature
# of class instances.
# However, in the world of tpcp-algorithms, thinks are significantly simpler than in the general case.
# This is because we know that the functionality of an algorithm is purely defined by its parameters.
# And we know, that the only side-effect an action method of an algorithm is allowed to have, is to write results to
# the `self` object.
# With this knowledge, we can implement a caching decorator that works for all action methods for all tpcp algorithms.
#
# It either can be applied as decorator to the class definition or called once with the class as an argument during
# runtime.
# The latter allows you to apply the caching to classes that you don't have control over.
#
# .. warning:: Depending on when and how you apply the decorator, it might not be correctly reapplied in the context
#    of multiprocessing.
#    Make sure to double-check that everything works as expected.
#    If not, you might be able to use :func:`~tpcp.parallel.register_global_parallel_callback` to fix the issue.
#    (at least in the context of joblib based multiprocessing).
#
# Below we demonstrate how to apply the decorator to a class after the fact.
from tpcp.caching import global_disk_cache, remove_any_cache

from examples.algorithms.algorithms_qrs_detection_final import QRSDetector

memory = Memory(HERE / ".cache", verbose=10)
global_disk_cache(memory)(QRSDetector)

# %%
# Now, if we call the QRS detector, we see that the cache is in the debug output.
# We load the example dataset here to demonstrate this.
from examples.datasets.datasets_final_ecg import ECGExampleData

example_data = ECGExampleData(
    HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
)
ecg_data = example_data[0].data["ecg"]

# %%
# As expected, we see that the algorithm was actually called twice (one for each config) and once from cache, even
# though, we created a completely new instance of the algorithm.
algo = QRSDetector()
algo = algo.detect(ecg_data, example_data.sampling_rate_hz)

algo2 = QRSDetector(max_heart_rate_bpm=180)
algo2 = algo2.detect(ecg_data, example_data.sampling_rate_hz)

# This one gets the result from cache
algo3 = QRSDetector()
algo3 = algo3.detect(ecg_data, example_data.sampling_rate_hz)

# %%
# This would allow us to globally patch the algorithm, without having to change any code in the algorithm itself.
# Read the documentation of the :func:`~tpcp.caching.global_disk_cache` for more information on how to configure the
# cache and some caveats of this approach.
#
# In this example, we remove the caching again, to not interfere with the rest of the example.
remove_any_cache(QRSDetector)
memory.clear()

# %%
# RAM Caching
# -----------
# Disk based caching makes sense if you want to reuse the cache between runs or across different processes.
# However, it can be comparatively slow.
# If you don't want to fill up your disk space and want fast access to a function result at multiple places in your
# code, memory/RAM caching is the way to go.
# Python provides a built-in decorator for memory caching, called
# `lru_cache <https://docs.python.org/3/library/functools.html#functools.lru_cache>`__.
# The ``lru_cache`` can be configured to store the last ``n`` function calls in memory.
# Like the joblib memory decorator, it caches the function output based on the function inputs.
# However, unlike the joblib memory decorator, it creates a new instance of a cache every time you apply it.
# Hence, the pattern of using a global function and caching it in a class-method does not work here.
#
# In general there are two approaches to use lru_cache:
#
# In, case you know how many function calls you want to cache (or you want to use an unlimited cache) and don't need to
# make this configurable, you can simply apply the decorator to your global function.
from functools import lru_cache


# This would cache the last 2 function calls.
# Passing `None` as the maxsize argument, will create an unlimited cache.
@lru_cache(maxsize=2)
def _get_data(participant_id: int):
    # Here we would do an expensive load a pre-processing step that takes a long time.
    # For this example, we just return a constant value that depends on the participant_id.

    # This print statement will only be executed when we don't cache the function.
    print(f"Un-cached call of `_get_data` with participant_id {participant_id}")

    return np.arange(10) * participant_id


class StaticMemoryCachedDataset(Dataset):
    @property
    def data(self):
        self.assert_is_single(None, "get_data")
        p_id = self.group_label.participant_id
        return _get_data(p_id)

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"participant_id": [1, 2, 3]})


# %%
# We can see that the cache works as expected, and we see only one print statement.
dataset = StaticMemoryCachedDataset()
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

# %%
# If we call the function with a different input, we see the print statement again.
dataset.get_subset(participant_id=2).data

# %%
# But going back to the previous input, we see the cached result again.
dataset.get_subset(participant_id=1).data

# %%
# This approach work well in cases, were you can easily predict how many function calls you want to cache and all the
# results are small enough that you don't run into memory issues.
# Datasets are again a great candidate, as you usually know, how many datapoints (and hence distinct function calls)
# you can expect.
# Hence, if the output data of all datapoints combined would easily fit into memory, you could set the cache size to
# the size of your dataset and benefit from fast access to the data, when accessed repeatedly in the same code.
# This is in particular important when you use the dataset in combination with methods like
# :func:`~tpcp.validate.cross_validate`,
#
# When all of your data would not fit into memory, caching expensive parts of the data loading with a ``maxsize=1`` can
# still sometimes be useful.
# For example, if the cached loading function does not just return one piece of information, but, for example, the
# sensor data and the reference data, you can cache the loading function and then split the data into two separate
# properties on your class.
# If users need to access both pieces of data, you avoid loading the data-file twice.
#
# Still, in many cases it beneficial to allow users to configure the cache size.
# This allows them to trade-off memory usage and performance.
# For example, if they test locally, they might want to use a smaller cache size more appropriate for their local
# machine, but when running on a server with more memory, they might want to increase the cache size to take advantage
# of the additional memory and potential performance gains.
# However, this can be tricky, as you need to keep a global reference to the cache instance and hence, can not easily
# create it within the class.
#
# The general problem we need to overcome is that we need to create the cache instance locally within the class, but
# need to make sure it is somehow persisted between different instances of the class or when the class instance is
# cloned.
#
# In the past we recommended to use a class variable to store the cache instance.
# However, since then we added :func:`~tpcp.caching.hybrid_cache`, which main purpose will be explained in the next
# section, but can already be used here to create a global cache instance.
# In the background it stores each cached function in a global registry and retrieves the cache instance from there,
# in case you wrap the function with the same parameters again.
#
# .. warning:: There is no magic implemented that clears this cache registry. This means you might store multiple caches
#    of functions that are not relevant anymore.
#    We explain in the next section, how you could handle this manually.
#
# For now, we will use staggered cache similar to how we use joblib Memory.
# We "re-wrap" the function we want to cache write before usage.
from tpcp.caching import hybrid_cache


def _get_data(participant_id: int):
    # Here we would do an expensive load a pre-processing step that takes a long time.
    # For this example, we just return a constant value that depends on the participant_id.

    # This print statement will only be executed when we don't cache the function.
    print(f"Un-cached call of `_get_data` with participant_id {participant_id}")

    return np.arange(10) * participant_id


class ConfigurableMemoryCachedDataset(Dataset):
    def __init__(
        self,
        # 0 is equivalent to no caching thanks to our helper function
        lru_cache_size: Optional[int] = 0,
        *,
        groupby_cols=None,
        subset_index=None,
    ) -> None:
        self.lru_cache_size = lru_cache_size
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _cached_get_data(self, participant_id: int):
        return hybrid_cache(lru_cache_maxsize=self.lru_cache_size)(_get_data)(
            participant_id
        )

    @property
    def data(self):
        self.assert_is_single(None, "get_data")
        p_id = self.group_label.participant_id
        return self._cached_get_data(p_id)

    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"participant_id": [1, 2, 3]})


# %%
# Now  we can see that in the un-cached form, we see the print statement every time we access the data attribute:
dataset = ConfigurableMemoryCachedDataset()
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

# %%
# We reset the cache
hybrid_cache.__cache_registry__.clear()

# %%
# When we configure the cache size to 1, we see the print statement only once, unless we change the input.
dataset = ConfigurableMemoryCachedDataset(lru_cache_size=1)
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

dataset.get_subset(participant_id=2).data

# %%
# And if we switch back to the previous input, we see that the func is run again.
dataset.get_subset(participant_id=1).data

# %%
# We reset the cache
hybrid_cache.__cache_registry__.clear()

# %%
# If we configure the cache size to larger values, we see that the print statement is only executed once per input.
dataset = ConfigurableMemoryCachedDataset(lru_cache_size=5)
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

dataset.get_subset(participant_id=2).data
dataset.get_subset(participant_id=1).data

# %%
# The cache will also stay consistent, if we clone the instance.
# Note, that there is no print statement.
dataset.clone().get_subset(participant_id=1).data

# %%
# Finally we clean the cache again, to not interfere with the rest of the example.
hybrid_cache.__cache_registry__.clear()

# %%
# This approach might also make sense for computations.
# In particular, when you have cases, where multiple algorithms might use the same pre-processing, you could use this
# approach to cache it.
# Or when performing a GridSearch, where only some parts of calculations are influenced by the changing parameters,
# caching could result in large performance increases.
#
# Within existing objects or full action cashing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Like with disk caching, we can also apply the memory caching to existing classes.
# The same caveats apply here, but using global RAM caching might even be more elegant to cache away repeated parts of
# your pipeline.
# For example, a filter that will be applied as a sub-step in multiple algorithms, could be cached globally.
#
# Below, we just repeat the example from above, but with RAM caching.
# Note that we set the size of the cache to 2, as we run the method below with two different configurations.
from tpcp.caching import global_ram_cache

global_ram_cache(2)(QRSDetector)

# %%
# Now, if we call the QRS detector, we see that the cache is working by inspecting the cache object.
algo = QRSDetector()
algo = algo.detect(ecg_data, example_data.sampling_rate_hz)

algo2 = QRSDetector(max_heart_rate_bpm=180)
algo2 = algo2.detect(ecg_data, example_data.sampling_rate_hz)

# This one gets the result from cache
algo3 = QRSDetector()
algo3 = algo3.detect(ecg_data, example_data.sampling_rate_hz)

# %%
# Now we can inspect the cache statistics.
from tpcp.caching import get_ram_cache_obj

cache_obj = get_ram_cache_obj(QRSDetector)
cache_obj.cache_info()

# %%
# Have a look at the documentation of :func:`~tpcp.caching.global_ram_cache` for more information.
#
# Again, we remove the caching again, to not interfere with the rest of the example.
remove_any_cache(QRSDetector)

# %%
# Hybrid Caching
# --------------
# Now that you have seen Disk and RAM caching, you might have noticed that both have their advantages and disadvantages.
# So, why not combine them?
# Basically, storing the results on disk and in RAM at the same time.
# Whenever, the fast RAM cache is available, we use it, but if the cache is not available, we fall back to the disk
# cache.
#
# This is exactly what we implemented in :func:`~tpcp.caching.hybrid_cache`.
# It is a decorator that takes a function and wraps it in a RAM cache and a disk cache.
#
# Below we define a simple function and wrap it with the staggered cache.
# Then we call it 3 times with different arguments.
from tpcp.caching import hybrid_cache


@hybrid_cache(Memory(".cache", verbose=10), lru_cache_maxsize=2)
def simple_func(a, b):
    print("This function was called without caching.")
    return a + b


simple_func(1, 2)
simple_func(2, 3)
simple_func(3, 4)

# %%
# Now the cache should contain all the results.
# However, as the lru cache has a size of two, it should only have the last two results, but the first one should be
# disk-cached.
# We can verify this, as we don't see any debug output when we rerun with these arguments, but when we
# call the function with the first argument again, we see the joblib-disk cache debug output.
# In all cases, we don't see the print statement from within the function indicating that the function was cached
# correctly.
#
# Calling again with the second and third argument:
simple_func(3, 4)
simple_func(2, 3)

# %%
# Now print output as expected
#
# If we call it with the first argument, we see the joblib-memory debug output, indicating that we hit the disk cache.
simple_func(1, 2)

# %%
# However, if we do that again now, the result should be stored in the lrucache again and we don't see any debug output.
simple_func(1, 2)
