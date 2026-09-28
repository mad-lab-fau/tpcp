Caching helper
==============

.. automodule:: tpcp.caching
    :no-members:
    :no-inherited-members:

Functions
---------

.. currentmodule:: tpcp.caching

.. autosummary::
    :toctree: generated/caching
    :template: function.rst

    hybrid_cache
    global_disk_cache
    global_ram_cache
    remove_disk_cache
    remove_ram_cache
    remove_any_cache
    get_ram_cache_obj

Faster caching for large inputs
-------------------------------

Enable ``fast_inaccurate_hashing=True`` when cached calls are still slow because
of large NumPy arrays or pandas DataFrames. It can reduce the time spent checking
whether a result is already cached.

Leave it disabled if cache lookups are already fast, or if you want to reuse
results cached with the default settings. Switching this option can require
results to be computed again.

.. code-block:: python

    from joblib import Memory
    from tpcp.caching import hybrid_cache

    @hybrid_cache(
        Memory(".cache", verbose=0),
        lru_cache_maxsize=2,
        fast_inaccurate_hashing=True,
    )
    def extract_features(data):
        return data.mean()
