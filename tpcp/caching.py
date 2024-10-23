"""Helper for caching related tasks."""

import binascii
import contextlib
import functools
import multiprocessing
import sys
import warnings
from collections.abc import Sequence
from pickle import PicklingError
from typing import Callable, Generic, Optional, TypeVar, Union

from joblib import Memory

from tpcp import Algorithm, get_action_methods_names, get_results, make_action_safe
from tpcp._hash import custom_hash
from tpcp.parallel import register_global_parallel_callback, remove_global_parallel_callback

_ALREADY_WARNED = False


def _global_cache_warning():
    global _ALREADY_WARNED  # noqa: PLW0603

    if multiprocessing.parent_process() is None and not _ALREADY_WARNED:
        # We want to avoid spamming the user with warnings if they are running multiple processes
        warnings.warn(
            "Global caching is a little tricky to get right and our implementation is not yet battle-tested. "
            "Please double check that the results are correct and report any issues you find.",
            UserWarning,
            stacklevel=3,
        )
        _ALREADY_WARNED = True


_instance_level_disk_cache_key = "__tpcp_disk_cached_action_method"
_class_level_lru_cache_key = "__tpcp_lru_cached_action_method"

T = TypeVar("T")


def _get_action_method(
    algorithm_object: type[Algorithm], action_func_name: Optional[str] = None
) -> tuple[str, Callable]:
    action_names = get_action_methods_names(algorithm_object)
    if len(action_names) > 1 and not action_func_name:
        raise ValueError(
            "When using cahcing on algorithm/pipeline objects with multiple action methods, you "
            "need to specify the action name during cache initialization."
        )
    if action_func_name and action_func_name not in action_names:
        raise ValueError(
            f"The action method {action_func_name} is not a valid action of the algorithm object. "
            f"Valid Actions: {action_names}"
        )
    primary_action_name = action_func_name if action_func_name else action_names[0]
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
    ... def add(
    ...     a: UniversalHashableWrapper[pd.DataFrame],
    ...     b: UniversalHashableWrapper[pd.DataFrame],
    ... ):
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


def _is_cached(obj, action_name):
    method = getattr(obj, action_name)
    cache_type = getattr(method, "__cache_type__", None)
    if cache_type is None:
        return False
    assert cache_type in ["disk", "ram"]
    return cache_type


def _handle_double_cached(obj, action_name, cache_type):
    is_cached = _is_cached(obj, action_name)
    if not is_cached:
        return False
    if is_cached == cache_type:
        warnings.warn(
            f"The action method {action_name} of {obj.__name__} is already cached using global {cache_type} "
            "cache. "
            "Repeated calls (even with changed settings have no effect)",
            stacklevel=2,
        )
        return True
    raise ValueError(
        f"The action method {action_name} of {obj.__name__} is already cached using global {is_cached}. "
        f"Caching it again using a global {cache_type} cache is not supported. "
        "Remove the global cache for the object first, before trying to cache it again."
    )


def _register_global_parallel_callback(func, name):
    def wrapper(_):
        func()

    def _callback():
        return None, wrapper

    register_global_parallel_callback(_callback, name=name)


def global_disk_cache(  # noqa: C901
    memory: Memory = Memory(None),
    *,
    cache_only: Optional[Sequence[str]] = None,
    action_method_name: Optional[str] = None,
    restore_in_parallel_process: bool = True,
):
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
    action_method_name
        In case the object you want to cache has multiple action methods, you can specify the name of the action method.
    restore_in_parallel_process
        If True, we will register a global parallel callback, that the diskcache will be correctly restored when using
        joblib parallel with the tpcp implementation of :func:`~tpcp.parallel.delayed`.

    Returns
    -------
    The algorithm class with the cached action method.

    See Also
    --------
    tpcp.caching.global_ram_cache
        Same as this function, but uses an LRU cache in RAM instead of a disk cache.
    """
    _global_cache_warning()

    def inner(algorithm_object: type[Algorithm]):  # noqa: C901
        if "<locals>" in algorithm_object.__qualname__:
            raise ValueError(
                "Global disk caching does not work with classes defined in inner scopes. "
                "The used classes need to be defined on the top level of a module or script."
            )

        # This only return the first action method, but this is fine for now
        # This method is "unbound", as we are working on the class, not an instance
        to_cache_action_method_name, action_method_raw = _get_action_method(algorithm_object, action_method_name)

        if _handle_double_cached(algorithm_object, to_cache_action_method_name, "disk"):
            return

        # Here we register a global callback, that caching is correctly restored in parallel processes using our
        # own "delayed" implementation.
        if restore_in_parallel_process:

            def recreate_cache():
                # Note, we need to check if the object is already cached to avoid the double cache warning
                if not _is_cached(algorithm_object, to_cache_action_method_name):
                    global_disk_cache(
                        memory,
                        cache_only=cache_only,
                        action_method_name=action_method_name,
                        restore_in_parallel_process=False,
                    )(algorithm_object)

            _register_global_parallel_callback(
                recreate_cache,
                name=f"global_disk_cache__{algorithm_object.__qualname__}__{to_cache_action_method_name}",
            )

        # We need to make the action method safe, as we are going to do weird stuff that expects correct implementation
        action_method = make_action_safe(action_method_raw)

        # our cached function gets the args, the kwargs and "fake self" as input.
        # Fake self is a clean clone of the algorithm instance.
        # It basically only encodes the parameters and the "name" of the algorithm.
        # We also pass the name of the action method to ensure that we record individual versions of the cache
        # for each action method.
        @memory.cache()
        def cached_action_method(__fake_self, __cache_only, __action_name, *args, **kwargs):
            after_action_instance: Algorithm = action_method(__fake_self, *args, **kwargs)
            # We return the results instead of the instance, as the results can be easily pickled.
            if __cache_only is None:
                return get_results(after_action_instance)
            return {k: v for k, v in get_results(after_action_instance).items() if k in __cache_only}

        @functools.wraps(action_method)
        def wrapper(self, *args_outer, **kwargs_outer):
            # We maintain a little cached function cache on each instance, to avoid recreating the wrapped func each
            # time the method is called.
            cache_store = getattr(self, _instance_level_disk_cache_key, {})
            if (cached_method := cache_store.get(to_cache_action_method_name, None)) is None:
                cache_store[to_cache_action_method_name] = cached_action_method
                cached_method = cached_action_method
                setattr(self, _instance_level_disk_cache_key, cache_store)

            new_instance = self.clone()
            current_type = type(new_instance)
            try:
                results = cached_method(
                    new_instance, cache_only, to_cache_action_method_name, *args_outer, **kwargs_outer
                )
            except PicklingError:
                if (
                    multiprocessing.parent_process() is None
                    or not getattr(current_type, "__module__", None) == "__main__"
                ):
                    raise

                # This is some black magic...
                # For some reason, if you defined your pipeline class in __main__, it can not be correctly pickled.
                # However, joblib memory needs to pickle it to hash it properly.
                # In a multi-process scenario, the class is send over to the subprocess using cloud pickle.
                # This means that the pickling issue can be circumvented there.
                # However, it is then not possible anymore to import the class in the subprocess, as __main__ is defined
                # differently.
                # To make it importable, and with that properly pickalable, we use the workaround below, that is already
                # used by joblib in other cases:
                # We manually add the obj to the new __main__ in the subprocess.
                modules_modified = []

                try:
                    name = current_type.__qualname__
                    to_add_obj = current_type
                except AttributeError:
                    name = current_type.__class__.__qualname__
                    to_add_obj = current_type.__class__
                mod = sys.modules["__main__"]
                if not hasattr(mod, name):
                    modules_modified.append((mod, name))
                    setattr(mod, name, to_add_obj)
                try:
                    results = cached_method(
                        new_instance, cache_only, to_cache_action_method_name, *args_outer, **kwargs_outer
                    )
                finally:
                    # Remove all new entries made to the main module.
                    for mod, name in modules_modified:
                        delattr(mod, name)

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        # This is used to restore the method and identify, if a method is cached.
        wrapper.__wrapped__ = action_method_raw
        wrapper.__cache_type__ = "disk"

        setattr(algorithm_object, to_cache_action_method_name, wrapper)

    return inner


def remove_disk_cache(algorithm_object: type[Algorithm]):
    """Remove the disk cache from an algorithm class.

    .. warning:: This only removes the cache for all future instances! Existing instances will still use the cache
    """
    for action_name in get_action_methods_names(algorithm_object):
        action_method = getattr(algorithm_object, action_name)
        if getattr(action_method, "__wrapped__", None) is not None:
            setattr(algorithm_object, action_name, action_method.__wrapped__)
            with contextlib.suppress(KeyError):
                remove_global_parallel_callback(f"global_disk_cache__{algorithm_object.__qualname__}__{action_name}")
    return algorithm_object


def global_ram_cache(  # noqa: C901
    max_n: Optional[int] = None,
    *,
    cache_only: Optional[Sequence[str]] = None,
    action_method_name: Optional[str] = None,
    restore_in_parallel_process: bool = True,
):
    """Wrap an algorithm/pipeline class to enable LRU based RAM cashing for the specified action method.

    .. warning:: When using this decorator, all actions calls are not made on the original object, but on a clone, and
       only the results are "re-attached" to the original object.
       In case you rely on side effects of the action method, this might cause issues.
       But, if you are relying on side effects, you are probably doing something wrong anyway.

    .. warning:: RAM cached objects can only be used with parallel processing, when the Algorithm/Pipeline class is
       defined NOT in the main module.
       Otherwise, you will get strange pickling errors.
       In general, using RAM cache with multi-processing does likely not make sense, as the RAM cache can not be shared
       between the individual processes.

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
    action_method_name
        In case the object you want to cache has multiple action methods, you can specify the name of the action method.
    restore_in_parallel_process
        If True, we will register a global parallel callback, that the diskcache will be correctly restored when using
        joblib parallel with the tpcp implementation of :func:`~tpcp.parallel.delayed`.

        .. warning:: This will only restore the cached setting, however, the actual cache is not shared and does not
                     carry over to the new process

    Returns
    -------
    The algorithm class with the cached action method.

    See Also
    --------
    tpcp.caching.global_disk_cache
        Same as this function, but uses a disk cache instead of an LRU cache in RAM.

    """
    _global_cache_warning()
    if cache_only is not None:
        cache_only = tuple(cache_only)

    def inner(algorithm_object: type[Algorithm]):
        to_cache_action_method_name, action_method_raw = _get_action_method(algorithm_object, action_method_name)

        if _handle_double_cached(algorithm_object, to_cache_action_method_name, "ram"):
            return algorithm_object

        # Here we register a global callback, that caching is correctly restored in parallel processes using our
        # own "delayed" implementation.
        if restore_in_parallel_process:

            def recreate_cache():
                # Note, we need to check if the object is already cached to avoid the double cache warning
                if not _is_cached(algorithm_object, to_cache_action_method_name):
                    global_ram_cache(
                        max_n=max_n,
                        cache_only=cache_only,
                        action_method_name=action_method_name,
                        restore_in_parallel_process=False,
                    )(algorithm_object)

            _register_global_parallel_callback(
                recreate_cache, name=f"global_ram_cache__{algorithm_object.__qualname__}__{to_cache_action_method_name}"
            )

        # We need to make the action method safe, as we are going to do weird stuff that expects correct implementation
        action_method = make_action_safe(action_method_raw)

        @functools.lru_cache(max_n)
        def cached_action(__fake_self, __cache_only, hashable_args, hashable_kwargs):
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

        @functools.wraps(action_method)
        def wrapper(self, *args_outer, **kwargs_outer):
            # Compared to the disk cache, the cash store is maintained on the class level to be shared between all
            # instances
            cache_store = getattr(algorithm_object, _class_level_lru_cache_key, {})
            if (cached_method := cache_store.get(to_cache_action_method_name, None)) is None:
                cache_store[to_cache_action_method_name] = cached_action
                cached_method = cached_action
                setattr(algorithm_object, _class_level_lru_cache_key, cache_store)
            results = cached_method(
                UniversalHashableWrapper(self.clone()),
                cache_only,
                UniversalHashableWrapper(args_outer),
                UniversalHashableWrapper(kwargs_outer),
            )

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        # This is used to restore the method and identify, if a method is cached.
        wrapper.__wrapped__ = action_method_raw
        wrapper.__cache_type__ = "ram"

        setattr(algorithm_object, to_cache_action_method_name, wrapper)
        return algorithm_object

    return inner


def remove_ram_cache(algorithm_object: type[Algorithm]):
    """Remove the RAM cache from an algorithm class."""
    cached_action_methods = getattr(algorithm_object, _class_level_lru_cache_key, {}).keys()

    for action_method_name in cached_action_methods:
        action_method = getattr(algorithm_object, action_method_name)
        if getattr(action_method, "__wrapped__", None) is not None:
            setattr(algorithm_object, action_method_name, action_method.__wrapped__)
            with contextlib.suppress(KeyError):
                remove_global_parallel_callback(
                    f"global_ram_cache__{algorithm_object.__qualname__}__{action_method_name}"
                )

    if getattr(algorithm_object, _class_level_lru_cache_key, None) is not None:
        delattr(algorithm_object, _class_level_lru_cache_key)

    return algorithm_object


def remove_any_cache(algorithm_object: type[Algorithm]):
    """Remove any cache from an algorithm class."""
    return remove_disk_cache(remove_ram_cache(algorithm_object))


def get_ram_cache_obj(algorithm_object: type[Algorithm], action_method_name: Optional[str] = None) -> Optional:
    """Get the RAM cache object from an algorithm class."""
    action_method_name, _ = _get_action_method(algorithm_object, action_method_name)
    return getattr(algorithm_object, _class_level_lru_cache_key, None)[action_method_name]


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
    _global_cache_warning()

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
