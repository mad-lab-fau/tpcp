"""Helper for caching related tasks."""
import binascii
import functools
import multiprocessing
import warnings
from collections.abc import Sequence
from typing import Callable, Generic, Optional, TypeVar, Union

from joblib import Memory

from tpcp import Algorithm, get_action_methods_names, get_results, make_action_safe
from tpcp._hash import custom_hash

if multiprocessing.parent_process() is None:
    # We want to avoid spamming the user with warnings if they are running multiple processes
    warnings.warn(
        "Global caching is a little tricky to get right and our implementation is not yet battle-tested. "
        "Please double check that the results are correct and report any issues you find.",
        UserWarning,
        stacklevel=2,
    )

_instance_level_disk_cache_key = "__tpcp_disk_cached_action_method"
_class_level_lru_cache_key = "__tpcp_lru_cached_action_method"

T = TypeVar("T")


def _get_primary_action_method(algorithm_object: type[Algorithm]):
    action_names = get_action_methods_names(algorithm_object)
    if len(action_names) > 1:
        raise NotImplementedError("Caching is only implemented for algorithms with a single action method.")
    primary_action_name = action_names[0]
    return primary_action_name, getattr(algorithm_object, primary_action_name)


class UniversalHashableWrapper(Generic[T]):
    """A wrapper that makes any object hashable.

    The primary use case is to make function arguments hashable so that they can be used with `functools.lru_cache`.
    Under the hood we use the same pickle based hashing developed for joblib and extended for tpcp to handle torch and
    tensorflow objects.


    Examples
    --------
    >>> import functools
    >>> from tpcp.caching import UniversalHashableWrapper
    >>> import pandas as pd
    >>>
    >>> @functools.lru_cache(maxsize=1)
    ... def add(a: UniversalHashableWrapper[pd.DataFrame], b: UniversalHashableWrapper[pd.DataFrame]):
    ...     return a.obj + b.obj
    >>> df1 = pd.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pd.DataFrame({"a": [4, 5, 6]})
    >>> add(UniversalHashableWrapper(df1), UniversalHashableWrapper(df2))

    """

    def __init__(self, obj: T) -> None:
        self.obj = obj

    def __hash__(self):
        """Hash the object using the pickle based approach."""
        return int(binascii.hexlify(custom_hash(self.obj).encode("utf-8")), 16)

    def __eq__(self, other):
        """Compare the object using their hash."""
        return custom_hash(self.obj) == custom_hash(other.obj)


def global_disk_cache(memory: Memory = Memory(None), *, cache_only: Optional[Sequence[str]] = None):
    """Wrap an algorithm/pipeline class to enable joblib based disk cashing for its primary action method.

    This will replace the action of the algorithm with a cached version.
    We use our knowledge about what is considered a "parameter" of an algorithm to clearly define the inputs of this
    cached function and avoid potential errors that might occur in a general caching solution.

    The cache will be invalidated, if any of the algorithm parameters change.

    This function can be used as a decorator on the class directly or called with the class as argument.
    In both cases the single global class object will be modified.
    At the moment there is no way of disabling the caching again.

    .. warning:: When using this decorator, all actions calls are not made on the original object, but on a clone, and
       only the results are "re-attached" to the original object.
       In case you rely on side effects of the action method, this might cause issues.
       But, if you are relying on side effects, you are probably doing something wrong anyway.

    Parameters
    ----------
    memory
        The joblib memory object that is used to cache the results.
        ``Memory(None)`` is equivalent to no caching.
    cache_only
        A list of strings that defines which results should be cached.
        If None, all results are cached.
        In case you only cash a subset of the results, only these results will be available on the returned objects.
        Also, during the first (uncached) call, only the results that are cached will be available.
        This might be helpful to reduce the size of the cache.

    Returns
    -------
    The algorithm class with the cached action method.

    See Also
    --------
    tpcp.caching.global_ram_cache
        Same as this function, but uses an LRU cache in RAM instead of a disk cache.
    """

    def inner(algorithm_object: type[Algorithm]):
        # This only return the first action method, but this is fine for now
        # This method is "unbound", as we are working on the class, not an instance
        primary_action_name, action_method = _get_primary_action_method(algorithm_object)

        # We need to make the action method safe, as we are going to do weird stuff that expects correct implementation
        action_method = make_action_safe(action_method)

        @functools.wraps(action_method)
        def wrapper(self, *args_outer, **kwargs_outer):
            if getattr(self, _instance_level_disk_cache_key, None) is None:
                # our cached function gets the args, the kwargs and "fake self" as input.
                # Fake self is a clean clone of the algorithm instance.
                # It basically only encodes the parameters and the "name" of the algorithm.
                def cachable_inner(__fake_self, __cache_only, *args, **kwargs):
                    after_action_instance: Algorithm = action_method(__fake_self, *args, **kwargs)
                    # We return the results instead of the instance, as the results can be easily pickled.
                    if __cache_only is None:
                        return get_results(after_action_instance)
                    return {k: v for k, v in get_results(after_action_instance).items() if k in __cache_only}

                cached_inner = memory.cache(cachable_inner)
                setattr(self, _instance_level_disk_cache_key, cached_inner)

            cached_inner = getattr(self, _instance_level_disk_cache_key)
            results = cached_inner(self.clone(), cache_only, *args_outer, **kwargs_outer)

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        wrapper.__wrapped__ = action_method

        setattr(algorithm_object, primary_action_name, wrapper)
        return algorithm_object

    return inner


def remove_disk_cache(algorithm_object: type[Algorithm]):
    """Remove the disk cache from an algorithm class."""
    primary_action_name, action_method = _get_primary_action_method(algorithm_object)

    if getattr(action_method, "__wrapped__", None) is not None:
        setattr(algorithm_object, primary_action_name, action_method.__wrapped__)

    return algorithm_object


def global_ram_cache(max_n: Optional[int] = None, *, cache_only: Optional[Sequence[str]] = None):
    """Wrap an algorithm/pipeline class to enable LRU based RAM cashing for its primary action method.

    .. warning:: When using this decorator, all actions calls are not made on the original object, but on a clone, and
       only the results are "re-attached" to the original object.
       In case you rely on side effects of the action method, this might cause issues.
       But, if you are relying on side effects, you are probably doing something wrong anyway.

    Parameters
    ----------
    max_n
        The maximum number of entries in the cache.
        If None, the cache will grow without limit.
    cache_only
        A list of strings that defines which results should be cached.
        If None, all results are cached.
        In case you only cash a subset of the results, only these results will be available on the returned objects.
        Also, during the first (uncached) call, only the results that are cached will be available.
        This might be helpful to reduce the size of the cache.

    Returns
    -------
    The algorithm class with the cached action method.

    See Also
    --------
    tpcp.caching.global_disk_cache
        Same as this function, but uses a disk cache instead of an LRU cache in RAM.

    """
    if cache_only is not None:
        cache_only = tuple(cache_only)

    def inner(algorithm_object: type[Algorithm]):
        primary_action_name, action_method = _get_primary_action_method(algorithm_object)

        # We need to make the action method safe, as we are going to do weird stuff that expects correct implementation
        action_method = make_action_safe(action_method)

        @functools.wraps(action_method)
        def wrapper(self, *args_outer, **kwargs_outer):
            if getattr(self, _class_level_lru_cache_key, None) is None:

                def cachable_inner(__fake_self, __cache_only, hashable_args, hashable_kwargs):
                    after_action_instance: Algorithm = action_method(
                        # Note: We need to clone the object here again.
                        # the LRU cache seems to calculate the hash of the object only after the function call.
                        # So we need to make sure that the object is not modified in the meantime.
                        __fake_self.obj.clone(),
                        *hashable_args.obj,
                        **hashable_kwargs.obj,
                    )
                    if __cache_only is None:
                        return get_results(after_action_instance)
                    return {k: v for k, v in get_results(after_action_instance).items() if k in __cache_only}

                # Create a LRUCache with max_n entries and store it on the class.
                # This way it is unique and shared between all instances of the class.
                cached_inner = {"cached_func": functools.lru_cache(max_n)(cachable_inner)}
                setattr(algorithm_object, _class_level_lru_cache_key, cached_inner)

            cached_inner = getattr(self, _class_level_lru_cache_key)

            results = cached_inner["cached_func"](
                UniversalHashableWrapper(self.clone()),
                cache_only,
                UniversalHashableWrapper(args_outer),
                UniversalHashableWrapper(kwargs_outer),
            )

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        wrapper.__wrapped__ = action_method

        setattr(algorithm_object, primary_action_name, wrapper)
        return algorithm_object

    return inner


def remove_ram_cache(algorithm_object: type[Algorithm]):
    """Remove the RAM cache from an algorithm class."""
    primary_action_name, action_method = _get_primary_action_method(algorithm_object)

    if getattr(action_method, "__wrapped__", None) is not None:
        setattr(algorithm_object, primary_action_name, action_method.__wrapped__)

    if getattr(algorithm_object, _class_level_lru_cache_key, None) is not None:
        delattr(algorithm_object, _class_level_lru_cache_key)

    return algorithm_object


def remove_any_cache(algorithm_object: type[Algorithm]):
    """Remove any cache from an algorithm class."""
    return remove_disk_cache(remove_ram_cache(algorithm_object))


def get_ram_cache_obj(algorithm_object: type[Algorithm]):
    """Get the RAM cache object from an algorithm class."""
    return getattr(algorithm_object, _class_level_lru_cache_key, None)["cached_func"]


_GLOBAL_CACHE_REGISTRY: dict[tuple[str, str], Callable] = {}


def hybrid_cache(
    joblib_memory: Memory = Memory(None),
    lru_cache_maxsize: Union[Optional[int], bool] = False,
):
    """Cache a function using joblib memory and a lru cache at the same time.

    This function attempts to be the best of both worlds and uses joblib.Memory to cache function calls between runs
    and a lru_cache to cache function calls during a run.

    When the cached function is called, the lookup will work as follows:

    1. Is the function result in the lrucache? If yes, return it.
    2. Is the function result in the joblib memory? If yes, return it and cache it in the lru cache.
    3. Call the function and cache it in the joblib memory and the lru cache. Return the result.

    It further solves one of the issues that you might run into with ``lru_cache``, that it is difficult to create a
    wrapped function during runtime, as calling ``lru_cache`` directly will create a new cache for each call.
    We work around this by using a global cache that stores the wrapped functions.
    The cache key is a tuple of the function name and a hash of the function, the joblib memory and the lru_cache paras.
    This means, if you create a new cache with different cache parameters, you will get a new cache, but if you call
    ``staggered_cache`` with the same parameters, you will get the same object back.

    You can access this global cache via the ``__cache_registry__`` attribute of this function
    (``staggered_cache.__cache_registry__``).

    Parameters
    ----------
    joblib_memory
        The joblib memory object that is used to cache the results.
        ``Memory(None)`` is equivalent to no caching.
    lru_cache_maxsize
        The maximum number of entries in the cache.
        If None, the cache will grow without limit.
        If False, no lru_cache is used.

    Returns
    -------
    caching_decorator
        A decorator that can be used to cache a function with the given parameters.

    Examples
    --------
    >>> import pandas as pd
    >>> from tpcp.caching import hybrid_cache
    >>> from joblib import Memory
    >>>
    >>> @hybrid_cache(Memory(".cache"), lru_cache_maxsize=1)
    ... def add(a: pd.DataFrame, b: pd.DataFrame):
    ...     return a + b
    >>> df1 = pd.DataFrame({"a": [1, 2, 3]})
    >>> df2 = pd.DataFrame({"a": [4, 5, 6]})
    >>> add(df1, df2)

    """

    def inner(function: Callable):
        paras_hash = custom_hash((function.__name__, id(function), joblib_memory, lru_cache_maxsize))
        cache_key = (function.__name__, paras_hash)
        if cache_key in _GLOBAL_CACHE_REGISTRY:
            return _GLOBAL_CACHE_REGISTRY[cache_key]

        if lru_cache_maxsize is False:

            @functools.wraps(function)
            def final_wrapped(*args, **kwargs):
                return joblib_memory.cache(function)(*args, **kwargs)
        else:

            def inner_cached(*hash_safe_args, **hash_safe_kwargs):
                args = tuple(arg.obj for arg in hash_safe_args)
                kwargs = {k: v.obj for k, v in hash_safe_kwargs.items()}
                return joblib_memory.cache(function)(*args, **kwargs)

            final_cached = functools.lru_cache(lru_cache_maxsize)(inner_cached)

            @functools.wraps(function)
            def final_wrapped(*args, **kwargs):
                hash_safe_args = tuple(UniversalHashableWrapper(arg) for arg in args)
                hash_safe_kwargs = {k: UniversalHashableWrapper(v) for k, v in kwargs.items()}
                return final_cached(*hash_safe_args, **hash_safe_kwargs)

        _GLOBAL_CACHE_REGISTRY[cache_key] = final_wrapped

        return final_wrapped

    return inner


hybrid_cache.__cache_registry__ = _GLOBAL_CACHE_REGISTRY


__all__ = [
    "global_disk_cache",
    "global_ram_cache",
    "UniversalHashableWrapper",
    "remove_disk_cache",
    "remove_ram_cache",
    "remove_any_cache",
    "get_ram_cache_obj",
    "hybrid_cache",
]
