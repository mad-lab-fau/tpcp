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

Fast best-effort hashing
-----------------------

``hybrid_cache`` retains its existing hashing by default. For large numeric inputs,
opt in to faster hashing for both RAM and disk lookups:

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

Fast mode uses XXH3-128 and reads all numeric values. Homogeneous numeric pandas
DataFrames are hashed by values, shape, dtypes, ordered columns, index, and
``attrs``; their internal memory layout is ignored. Other objects, including
object and extension dtypes, use the existing serialization with the faster
digest. NumPy arrays retain joblib's dtype, shape, and stride handling.

This is a best-effort content check. Unusual representation or metadata differences
may be missed and can lead to an incorrect cache hit. Use the default when those
differences matter to the cached function. Fast disk entries have their own key
version and do not reuse default entries. Joblib still checks the original
function's code for disk-cache invalidation.

Fast RAM keys are computed once per argument on each call, so mutations between
calls cause a new lookup. A RAM miss hashes the arguments again for disk lookup.
As with ordinary caching, avoid modifying arguments while a call is running or
mutating cached return values. RAM entries retain the existing cache lifecycle;
clear/recreate RAM caches when changing the function implementation.

Algorithm safety checks use :func:`tpcp.misc.custom_hash`, which now enables fast
hashing automatically. Its default digest values have changed. Pass
``hash_name="md5"`` explicitly when comparing against fingerprints from the old
implementation; explicit ``hash_name="sha1"`` also retains its previous behavior.
