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
   This means, this cache is only usefull, if the same computation result is accessed multiple times within the same
   process/script.
   Also, your RAM space is usually much more limited than your disk space, so you need to be careful to not cache too
   much data.

Disk Caching
------------

The easiest way to perform disk-caching in Python is to use the `joblib <https://joblib.readthedocs.io/en/latest/>`__
library.
We highly recommend to read through their documentation, as it provides a genral useful information on that topic.
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
To make this location parameter accessible to the user of your tpcp object, we recommend to add a `memory` parameter
to your `init` method, that can take a joblib memory instance.

Below, you can see an example Dataset that uses this pattern.
Note, that we factored out the processing that we want to cache into a global pure function and then cache this function
every time we call the data attribute.
"""
import warnings
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
HERE = Path()

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
# Memory Caching
# --------------
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
# However, to allow this we need to write some additional tooling.
#
# The general problem we need to overcome is that we need to create the cache instance locally within the class, but
# need to make sure it is somehow persisted between different instances of the class or when the class instance is
# cloned.
# To do this, we can use a class attribute to store the cache instance.


def get_func_from_class_cache(cls, cache_key, maxsize, func):
    """Set the cache for a class.

    Parameters
    ----------
    cls
        The class to set the cache for.
    cache_key
        The key to use for the cache.
    maxsize
        The maximum size of the cache.
    func
        The function to cache.

    """
    class_cache = getattr(cls, "__CACHE", None)
    if not class_cache:
        class_cache = {}
        setattr(cls, "__CACHE", class_cache)
    if cached_func := class_cache.get(cache_key, None):
        if cached_func.cache_info().maxsize != maxsize:
            warnings.warn(
                f"There already exists a cached function for {cache_key}, but with a different `maxsize` parameter. "
                "We will ignore the new maxsize parameter and reuse the old cache"
            )
        return cached_func
    if maxsize == 0:
        return func
    class_cache[cache_key] = lru_cache(maxsize=maxsize)(func)
    return class_cache[cache_key]


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
        return get_func_from_class_cache(
            self.__class__, cache_key="get_data", maxsize=self.lru_cache_size, func=_get_data
        )(participant_id)

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
# We reset the class cache
ConfigurableMemoryCachedDataset.__CACHE = {}

# %%
# When we configure the cache size to 1, we see the print statement only once, unless we change the input.
dataset = ConfigurableMemoryCachedDataset(lru_cache_size=1)
dataset.get_subset(participant_id=1).data
dataset.get_subset(participant_id=1).data

dataset.get_subset(participant_id=2).data

# %%
# And if we switch back to the previous input, we see that the func is run again.
dataset.get_subset(participant_id=1).data

ConfigurableMemoryCachedDataset.__CACHE = {}

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
# If you want to use this approach in your own code, you can copy the helper function from above.
#
# This approach might also make sense for computations.
# In particular, when you have cases, where multiple algorithms might use the same pre-processing, you could use this
# approach to cache it.
# Or when performing a GridSearch, where only some parts of calculations are influenced by the changing parameters,
# caching could result in large performance increases.
