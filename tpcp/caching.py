import functools
from typing import Type, Optional, Sequence

from joblib import Memory

from tpcp import Algorithm, get_action_methods_names, get_results

instance_level_cache_key = "__tpcp_cached_action_method"

def global_disk_cache(memory: Memory = Memory(None), *, cache_only: Optional[Sequence[str]] = None):
    """A decorator to wrap an algorithm with a global disk cache.

    This will replace the action of the algorithm with a cached version.
    We use our knowledge about what is considered a "parameter" of an algorithm to clearly define the inputs of this
    cached function and avoid potential errors that might occur in a general caching solution.
    """
    def inner(algorithm_object: Type[Algorithm]):
        # This only return the first action method, but this is fine for now
        # This method is "unbound", as we are working on the class, not an instance

        action_names = get_action_methods_names(algorithm_object)
        if len(action_names) > 1:
            raise NotImplementedError("Caching is only implemented for algorithms with a single action method.")
        primary_action_name = action_names[0]
        action_method = getattr(algorithm_object, primary_action_name)

        @functools.wraps(action_method)
        def wrapped(self, *args_outer, **kwargs_outer):
            params = self.get_params()

            if getattr(self, instance_level_cache_key, None) is None:
                # our cached function gets the args, the kwargs and the params as input
                # We now that for a "proper" algorithm, the params are the only thing that can change the results.
                # Having them as input will invalidate the cache, if the params change.
                def cachable_inner(__params, __cache_only, *args, **kwargs):
                    after_action_instance: Algorithm = action_method(self, *args, **kwargs)
                    # We return the results instead of the instance, as the results can be easily pickled.
                    return {k: v for k, v in get_results(after_action_instance).items() if k in __cache_only}


                cached_inner = memory.cache(cachable_inner)
                setattr(self, instance_level_cache_key, cached_inner)

            cached_inner = getattr(self, instance_level_cache_key)
            results = cached_inner(params, cache_only, *args_outer, **kwargs_outer)

            # manually "glue" the results back to the instance
            for result_name, result in results.items():
                setattr(self, result_name, result)

            return self

        setattr(algorithm_object, primary_action_name, wrapped)
        return algorithm_object

    return inner

