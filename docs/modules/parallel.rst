tpcp.parallel: Context-aware parallel execution
================================================

For practical caveats around joblib-based multiprocessing, worker state, serialization, and import costs, see
:ref:`multiprocessing_caveats`.

.. automodule:: tpcp.parallel
    :no-members:
    :no-inherited-members:

Functions
---------

.. currentmodule:: tpcp.parallel

.. autosummary::
    :toctree: generated/parallel
    :template: function.rst

    delayed
    execution_context
    run_in_worker
    job_metadata
    get_job_metadata
    get_worker_context
    register_global_parallel_callback
    register_parallel_side_channel
    remove_global_parallel_callback
    remove_parallel_side_channel

Classes
-------

.. autosummary::
    :toctree: generated/parallel
    :template: class.rst

    Parallel
    Task
    TaskConfig
    Job
    JobEvent
    WorkerContext

Experimental async execution
----------------------------

``tpcp.aio.validate.cross_validate`` and ``tpcp.aio.optimize.optimize`` return
lazy, awaitable bound tasks. Inspect a task before execution, await it directly,
or pass it to ``Worker.execute``. Arguments are bound by reference, as with
``joblib.delayed``. For ``asyncio.create_task`` or ``TaskGroup.create_task``,
wrap the await in a coroutine.

.. code-block:: python

    from tpcp.aio.parallel import Task, execution_context, get_worker_context
    from tpcp.aio.validate import cross_validate

    class Algorithm:
        def async_optimize(self, data):
            # The callee owns the resource request, not its caller.
            return Task(self.optimize, resource="gpu")(data)

        def optimize(self, data):
            worker = get_worker_context()
            # Select worker.gpu_id in your GPU framework here.
            return compute(data)

    async def evaluate(search, dataset, scoring):
        effect = cross_validate(search, dataset, scoring=scoring, cv=5)
        assert effect.config.kind == "workflow"
        async with execution_context(n_jobs=4, n_gpu_jobs=1, batch_size="auto"):
            return await effect

Worker tasks run synchronous callables in joblib's reusable loky process pool.
``Task(async_function)(...)`` instead awaits an async callable on the
main event loop. Workflow tasks compose child effects without reserving a worker;
only leaf effects occupy slots. ``asyncio.gather`` can await independent effects
concurrently. A GPU search can override ``async_optimize`` to return
``super().async_optimize(dataset, resource="gpu")``; the surrounding CV awaits
its effect normally. An optimizer effect may return a new fitted instance, for
example after process serialization. CV uses the returned instance for held-out
scoring and ``return_optimizer``. ``resource="gpu"`` on CV is also available as a convenience
for its candidate and held-out scoring jobs.

``n_jobs`` is a positive CPU/process budget. ``n_gpu_jobs`` is an independent
nonnegative GPU budget; each GPU batch also occupies a CPU/process slot. Waiting
for a GPU does not reserve a CPU worker. ``gpu_ids=(0, 1)`` assigns devices to GPU
slots (by default IDs run from zero to ``n_gpu_jobs - 1``). Device placement is
performed by your callable. These slots coordinate this execution context;
they do not coordinate unrelated programs using the same device.

The adapter uses joblib's native ``BatchedCalls``, cloudpickle, automatic array
memmapping and reusable loky executor. ``batch_size`` accepts a positive integer
or ``"auto"``; automatic tuning uses joblib's batch timings, independently for
CPU and GPU queues. A batch runs its jobs serially under one resource reservation.
``n_jobs`` therefore limits concurrent batches, rather than their total number
of contained jobs. Worker exceptions are returned separately for each job in a
batch. Pass serialization settings such as ``max_nbytes``, ``mmap_mode`` and
``temp_folder`` to ``execution_context``. Private joblib interfaces are isolated
in the experimental adapter and require compatibility checks when joblib changes.

Metadata and worker state
~~~~~~~~~~~~~~~~~~~~~~~~~

``job_metadata(cv_fold=2)`` attaches metadata to nested tasks through a context
variable; ``Task(..., metadata={"stage": "train"})`` adds per-call metadata.
Read it with ``get_job_metadata()`` in the workflow or worker. GridSearch/CV
include fold, parameter-candidate, parameter values and scoring-stage metadata.
``get_worker_context()`` additionally returns worker name, PID, assigned GPU ID,
job ID and batch ID/index/size. With ``Task(..., pass_context=True)``, the callable
receives this ``WorkerContext`` as its first positional argument. Context is
restored for every job and reset afterward, including between jobs in a batch.
Existing TPCP worker-state callbacks and result side channels are preserved.

``execution_context(on_job_event=callback)`` delivers queued, started, completed,
failed and cancelled notifications on the main event-loop thread. Successful
completion and worker failures include the worker context. Started means a job
was admitted for batch preparation; it does not mean the process has begun that job.
The callback can update custom progress displays. Built-in tqdm bars report
completed CV folds and candidate combinations; inner bars use fold labels.

Lifecycle and prototype limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Asyncio tasks inherit the execution context. Overlapping contexts must use the
same event loop and identical configuration, sharing one adapter and slot budget.
The last context to exit drains accepted calls, including queued calls, then
cleans its memmap files and leaves the pool available for subsequent joblib calls.
Cancellation does not interrupt running processes. Context exit waits without
blocking the event loop, and propagates cancellation after drain and cleanup.

Results and search selection are applied on the main thread. Timings include
scheduler waits and process startup. Task-group failures raise ``ExceptionGroup``.
Synchronous search/scorer ``n_jobs`` and ``pre_dispatch`` settings do not apply;
datapoints run serially inside each scoring job. Custom worker functions must
also avoid spawning pools. This prototype flattens explicitly composed async
workflows; it does not transform arbitrary synchronous optimizers or user code.
For best responsiveness with long jobs, use ``batch_size=1``.

See ``examples/validation/_05_async_grid_search.py`` for a runnable example.

Sharing task results
~~~~~~~~~~~~~~~~~~~~

A Worker uses a filesystem joblib Memory for artifacts. Pass the same
InMemoryTaskStorage instance to share records in one process. For independent
processes on one machine, give each participant the same SQLite file and Memory
directory. A multi-machine deployment would require a server database.

.. code-block:: python

    from joblib import Memory
    from tpcp.aio.parallel import CachedTask, SQLiteTaskStorage, Task, Worker

    async def evaluate_fold(dataset_revision, fold_id, data):
        memory = Memory("artifacts", verbose=0)
        async with Worker(memory, SQLiteTaskStorage("tasks.sqlite"), n_jobs=2) as worker:
            with worker.scope(("study-2026", dataset_revision)):
                async with worker.subtask("fold", key=fold_id) as fold:
                    if fold.acquired:
                        features = await CachedTask(transform)(data)
                        scores = await Task(score)(features)
                        await fold.complete(scores)
                    else:
                        scores = await fold.result()
                    return scores

Preparation captures the current manual scope and resolves the arguments once.
Keep bound inputs stable until completion. Nested scope keys accumulate. At
the root all participants share an empty scope. Use distinct experiment keys
or seeds for independent random trials. Ordinary tasks replay their first
successful result in that scope. Scope exit retains child records and artifacts.
CachedTask uses the original function's joblib cache entry across all scopes.
Participants must use consistent function definitions, dependencies and inputs.
Ordinary joblib callers can read and populate the same cache without taking
worker leases.

For explicit scheduling, call worker.prepare(task) followed by
worker.acquire(prepared) when capacity is available. Inspect
decision.record for the attempt, status, timestamps and artifact reference.
An acquired claim needs await worker.run(decision). For an existing claim,
choose other work or await worker.result(decision). If its owner fails or
its lease expires, acquire again. If the enclosing subtask expired, re-enter
that subtask and prepare its children under the new claim. A stalled owner can
duplicate computation after takeover. Publication checks the ordinary task
attempt and all enclosing leases. CachedTask checks its independent lease.

Earlier experimental examples bound arguments using Task(fn, args=(x,)).
Use Task(fn)(x). Put scheduling options on the first call and function
arguments on the second.
