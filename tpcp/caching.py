import functools
from collections.abc import Sequence
from typing import Optional

import binascii
from joblib import Memory

from tpcp import Algorithm, get_action_methods_names, get_results, make_action_safe
from tpcp._hash import custom_hash

instance_level_disk_cache_key = "__tpcp_disk_cached_action_method"
class_level_lru_cache_key = "__tpcp_lru_cached_action_method"


class UniversalHashableWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return int(binascii.hexlify(custom_hash(self.obj).encode("utf-8")), 16)

    def __eq__(self, other):
        return custom_hash(self.obj) == custom_hash(other.obj)


def global_disk_cache(
    memory: Memory = Memory(None), *, cache_only: Optional[Sequence[str]] = None
):
    """A decorator to wrap an algorithm with a global disk cache.

    This will replace the action of the algorithm with a cached version.
    We use our knowledge about what is considered a "parameter" of an algorithm to clearly define the inputs of this
    cached function and avoid potential errors that might occur in a general caching solution.
    """

    def inner(algorithm_object: type[Algorithm]):
        # This only return the first action method, but this is fine for now
        # This method is "unbound", as we are working on the class, not an instance

        action_names = get_action_methods_names(algorithm_object)
        if len(action_names) > 1:
            raise NotImplementedError(
                "Caching is only implemented for algorithms with a single action method."
            )
        primary_action_name = action_names[0]
        action_method = getattr(algorithm_object, primary_action_name)

        @functools.wraps(action_method)
        def wrapped(self, *args_outer, **kwargs_outer):
            if getattr(self, instance_level_disk_cache_key, None) is None:
                # our cached function gets the args, the kwargs and "fake self" as input.
                # Fake self is a clean clone of the algorithm instance.
                # It basically only encodes the parameters and the "name" of the algorithm.
                def cachable_inner(__fake_self, __cache_only, *args, **kwargs):
                    after_action_instance: Algorithm = make_action_safe(action_method)(
                        self, *args, **kwargs
                    )
                    # We return the results instead of the instance, as the results can be easily pickled.
                    if __cache_only is None:
                        return get_results(after_action_instance)
                    return {
                        k: v
                        for k, v in get_results(after_action_instance).items()
                        if k in __cache_only
                    }

                cached_inner = memory.cache(cachable_inner)
                setattr(self, instance_level_disk_cache_key, cached_inner)

            cached_inner = getattr(self, instance_level_disk_cache_key)
            results = cached_inner(
                self.clone(), cache_only, *args_outer, **kwargs_outer
            )

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        setattr(algorithm_object, primary_action_name, wrapped)
        return algorithm_object

    return inner


def global_ram_cache(max_n: Optional[int] = None, *, cache_only: Optional[Sequence[str]] = None):
    def inner(algorithm_object: type[Algorithm]):
        action_names = get_action_methods_names(algorithm_object)
        if len(action_names) > 1:
            raise NotImplementedError(
                "Caching is only implemented for algorithms with a single action method."
            )
        primary_action_name = action_names[0]
        action_method = getattr(algorithm_object, primary_action_name)

        @functools.wraps(action_method)
        def wrapped(self, *args_outer, **kwargs_outer):
            if getattr(self, class_level_lru_cache_key, None) is None:

                def cachable_inner(
                    __fake_self, __cache_only, hashable_args, hashable_kwargs
                ):
                    after_action_instance: Algorithm = make_action_safe(action_method)(
                        self, *hashable_args.obj, **hashable_kwargs.obj
                    )
                    if __cache_only is None:
                        return get_results(after_action_instance)
                    return {
                        k: v
                        for k, v in get_results(after_action_instance).items()
                        if k in __cache_only
                    }

                # Create a LRUCache with max_n entries and store it on the class.
                # This way it is unique and shared between all instances of the class.
                cached_inner = {"cached_func" : functools.lru_cache(max_n)(cachable_inner)}
                setattr(algorithm_object, class_level_lru_cache_key, cached_inner)

            cached_inner = getattr(self, class_level_lru_cache_key)

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

        setattr(algorithm_object, primary_action_name, wrapped)
        return algorithm_object

    return inner
