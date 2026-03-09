.. _multiprocessing_caveats:

Multiprocessing Caveats
=======================

`tpcp` uses `joblib <https://joblib.readthedocs.io/>`_ for multiprocessing in validation and optimization helpers.
This page is intended to be the durable documentation version of the multiprocessing notes that were previously tracked
in GitHub issue `#119 <https://github.com/mad-lab-fau/tpcp/issues/119>`_.

The main point is that most multiprocessing problems in `tpcp` are not specific to `tpcp` itself.
They come from Python process semantics, joblib worker reuse, and serialization.
However, these issues surface frequently in `tpcp` because validation, scoring, caching, and optimization often run
many jobs in parallel and often rely on runtime configuration.

When this matters
-----------------

These caveats matter whenever you use joblib-based multiprocessing, for example via:

- :func:`~tpcp.validate.validate`
- :func:`~tpcp.validate.cross_validate`
- :class:`~tpcp.validate.Scorer`
- optimizers that expose `n_jobs`
- your own joblib parallel code that reuses :mod:`tpcp.parallel`

Global variables and runtime modifications
------------------------------------------

In child processes, global variables are reset to their import-time values.
This is expected: the code that mutated those globals in the parent process is not replayed automatically in workers.

This becomes relevant whenever your runtime behavior depends on global state, for example:

- sklearn or pandas global configuration
- global registries
- cache decorators or other runtime patching
- monkey-patching a class or function after import

In practice this means that code can work in the parent process and then behave differently in workers without any
local code changes.

This is also relevant beyond `tpcp`.
For example, scikit-learn has its own workaround for some of its internal `Parallel` usage, but those fixes do not
automatically apply to arbitrary user-defined multiprocessing or to tpcp-specific wrappers.

tpcp workaround: ``tpcp.parallel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`tpcp` provides :mod:`tpcp.parallel` as a workaround for this class of problem.
Its :func:`~tpcp.parallel.delayed` wrapper captures registered state in the parent process and restores it in workers.

The workflow is:

1. register a callback with :func:`~tpcp.parallel.register_global_parallel_callback`
2. the callback returns a ``(value, setter)`` pair
3. `tpcp.parallel.delayed` captures the value in the parent process
4. the setter is called in the worker before the actual function executes

This is useful whenever some global configuration or global runtime setup must be visible in worker processes.

Example:

.. code-block:: python

    from joblib import Parallel
    from sklearn import get_config, set_config
    from tpcp.parallel import delayed, register_global_parallel_callback, remove_global_parallel_callback

    def callback():
        def setter(config):
            set_config(**config)

        return get_config(), setter

    name = register_global_parallel_callback(callback)
    try:
        Parallel(n_jobs=2)(delayed(worker_func)() for _ in range(2))
    finally:
        remove_global_parallel_callback(name)

Related background:

- `joblib issue #1071 <https://github.com/joblib/joblib/issues/1071>`_
- `scikit-learn PR #17634 <https://github.com/scikit-learn/scikit-learn/pull/17634>`_

.. note::
    The workaround is only applied when `tpcp.parallel.delayed` is used.
    If your own code uses `joblib.delayed` directly, registered callbacks will not run.

.. warning::
    Different libraries may implement their own worker-state restoration logic.
    These fixes are not guaranteed to be compatible with each other without additional configuration.

Process pools are reused
------------------------

When using joblib's default `loky` backend, worker processes are often reused.
This happens not only within a single `Parallel(...)` call, but can also happen across later parallel calls.

This has an important consequence:
if a worker mutates global state, that modified state can still be present when a later job is executed in the same
worker.

This can lead to surprising behavior such as:

- state "leaking" between logically independent parallel jobs
- nondeterministic test failures that depend on execution order
- interactions with the global-state workaround above that are hard to reason about

If you need a clean pool, for example in tests, shut the reusable executor down explicitly:

.. code-block:: python

    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True, kill_workers=True)

This is mainly useful in tests or in debugging scenarios where you need to eliminate worker reuse as a variable.

Serialization is often the real problem
---------------------------------------

Many multiprocessing failures are actually serialization failures.
To execute work in child processes, joblib must serialize the function to call and all relevant inputs.

For custom classes, functions, and closures this can become subtle.

At a high level, standard pickle prefers passing objects by reference:

- for global objects, it stores import path + object name
- for instances, it stores the parent type plus object state

This creates a number of common failure modes.

Common serialization traps
~~~~~~~~~~~~~~~~~~~~~~~~~~

The most common problematic cases are:

1. instances of classes whose type is not defined globally
2. classes defined inside functions
3. lambdas or other callables without a stable global reference
4. functions or classes that only exist in `__main__`
5. globally defined objects that are replaced or modified at runtime before serialization

The `__main__` case is particularly common and confusing.
Objects defined in the currently executing script do not have a stable import path from the point of view of worker
processes.

Typical fixes
~~~~~~~~~~~~~

If you see serialization failures:

- if the error mentions `__main__`, move the relevant class/function to an importable module
- if objects are created dynamically from runtime config, move the object creation into the worker and only pass the
  config
- if the object depends on runtime patching, restore that patch explicitly in workers using
  :func:`~tpcp.parallel.register_global_parallel_callback`

About ``cloudpickle``
~~~~~~~~~~~~~~~~~~~~~

`joblib` often falls back to `cloudpickle`, which can serialize many objects that plain pickle cannot.
This helps with many dynamic objects and closures.

However:

- it is slower
- it can hide the underlying reason why serialization is fragile
- when it also fails, the error messages can become harder to interpret

So while `cloudpickle` is often helpful, it should not be treated as a guarantee that any dynamic runtime structure is
safe for multiprocessing.

Runtime patching and caching
----------------------------

Runtime patching deserves separate attention because it is common in `tpcp`.
For example, you might apply caching decorators or other wrappers after import or after class definition.

This is convenient, but multiprocessing changes the situation:

- workers may import the original object, not the patched one
- runtime changes in the parent process may not be replayed in workers
- if worker pools are reused, worker-local patched state may persist longer than expected

This is particularly relevant when using tpcp caching utilities with multiprocessing.
If a runtime-applied cache or decorator must also exist in workers, do not assume that parent-side setup is enough.
Use the documented restoration mechanisms and verify the behavior in parallel explicitly.

The caching recipe already hints at this caveat:

- `examples/recipies/_01_caching.py`

Imports can dominate runtime
----------------------------

Because objects are reconstructed in worker processes, all relevant imports must also resolve correctly in workers.

This can have a substantial runtime cost.
For heavy optional dependencies, import overhead can dominate the actual work.

TensorFlow is a typical example: importing it can take seconds, which can erase much of the benefit of
multiprocessing for smaller tasks.

A useful rule is:

- if a dependency is optional and only needed in some code paths, delay the import until it is actually needed

This does not solve all multiprocessing issues, but it can materially improve runtime behavior.

Practical debugging checklist
-----------------------------

If multiprocessing behaves strangely, check in this order:

1. Is the problem actually missing global runtime state in workers?
2. Does the failure mention pickling, `__main__`, or an import path?
3. Are you depending on runtime patching or monkey-patching?
4. Could worker reuse be leaking state between jobs?
5. Is import overhead larger than the actual parallel workload?

In many cases, the fastest path to clarity is to temporarily run with `n_jobs=1`.
If the problem disappears, focus on worker state and serialization next.

tpcp-specific takeaways
-----------------------

- If global configuration is missing in workers, use :mod:`tpcp.parallel`.
- If you use custom parallel code together with `tpcp` worker-state restoration, make sure you use
  :func:`~tpcp.parallel.delayed`, not `joblib.delayed`.
- If tests depend on clean workers, explicitly shut down the reusable `loky` executor.
- If a pickle error mentions `__main__`, move the relevant object to a normal module first.
- If runtime-created objects are difficult to serialize, pass configuration into workers and build the object there.
- If the multiprocessing setup becomes too fragile, `n_jobs=1` is often the pragmatic fallback.

Related APIs
------------

- :mod:`tpcp.parallel`
- :func:`tpcp.parallel.delayed`
- :func:`tpcp.parallel.register_global_parallel_callback`
- :func:`tpcp.parallel.remove_global_parallel_callback`
