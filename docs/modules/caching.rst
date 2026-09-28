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

Choosing a hashing mode
-----------------------

``hybrid_cache`` identifies cached calls by hashing their arguments. The default
mode uses MD5 with pickle-based object traversal, including pandas storage details.
For large numeric inputs, set ``fast_inaccurate_hashing=True`` to use faster,
best-effort content hashing for both RAM and disk lookups:

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

Both modes use pickle-based object traversal with joblib's direct buffer handling
for numeric NumPy arrays. Fast mode uses XXH3-128 instead of MD5 and reads all
values without sampling. Arrays and DataFrames keep their storage representation;
changing between row-major and column-major layout can produce a different key.

XXH3-128 is a non-cryptographic hash suited to fast content checks on trusted data.
Hashes are a best-effort check, not a guarantee of object equality. The two modes
use separate disk cache entries.

Cache lifetime and mutations
----------------------------

Fast RAM keys are computed once per argument on each call. Changes to hashed
values or metadata between calls cause a new lookup. A RAM miss hashes the
arguments again for disk lookup. Avoid modifying arguments while a call is running
or mutating cached return values.

Joblib checks the function's code when looking up disk entries and invalidates
results when that code changes. Clear and recreate RAM caches after changing the
function implementation.

Hashing for safety checks
-------------------------

Algorithm safety checks use :func:`tpcp.misc.custom_hash`, which uses fast,
best-effort hashing by default. To select MD5 or SHA1 with the same object
traversal, pass
``hash_name="md5"`` or ``hash_name="sha1"``.
