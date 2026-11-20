tpcp.validate: Validation helper
================================

.. automodule:: tpcp.validate
    :no-members:
    :no-inherited-members:

Classes
-------

.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: class_with_private.rst

    BaseDatasetSplitter
    CombinedSplitter
    DatasetSplitter
    NoSplit
    SubsetSplitter

Scoring
-------
.. currentmodule:: tpcp.validate


.. autosummary::
   :toctree: generated/validate
   :template: class_with_private.rst

    Scorer
    Aggregator
    FloatAggregator
    MacroFloatAggregator

.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: function.rst

    mean_agg
    no_agg


Functions
---------
.. currentmodule:: tpcp.validate

.. autosummary::
   :toctree: generated/validate
   :template: function.rst

    cross_validate
    validate

The :doc:`scoring aliases <types>` describe score functions, aggregator return values, and the
``scoring`` argument accepted by validation functions.

Async scoring and pipeline actions
----------------------------------

``tpcp.aio.validate.Scorer`` submits one worker task per datapoint. It calls the
score function with an independent pipeline clone, then runs callbacks and
aggregation on the coordinator in dataset order. The async validate function
returns the same score fields and group labels as synchronous validation.

.. code-block:: python

    from tpcp.aio import execution_context
    from tpcp.aio.validate import Scorer, validate

    async def evaluate(pipeline, dataset, score_one):
        async with execution_context(n_jobs=4):
            scorer = Scorer(score_one, progress_bar=False)
            return await validate(pipeline, dataset, scoring=scorer)

Built-in scorers and search objectives call synchronous score functions with
synchronous pipelines. To compose native async actions, subclass ``Algorithm``,
``Pipeline`` or ``OptimizablePipeline`` from tpcp.aio. Declare action names on an
algorithm with _action_methods. Pipelines already declare ``safe_run`` and run.
Bind a coordinator method with ``self.action`` and await worker tasks inside it.

.. code-block:: python

    from tpcp.aio import Pipeline, Task

    class Detector(Pipeline):
        def run(self, datapoint):
            return self.action(self._detect, datapoint)

        async def _detect(self, datapoint):
            self.events_ = await Task(find_events)(datapoint)
            return self

Await ``detector.safe_run(datapoint)`` to apply pipeline action checks. Implement
``self_optimize`` or ``self_optimize_with_info`` on an async optimizable pipeline to
bind custom training. Call native async actions directly inside your workflow.
The built-in ``Optimize`` wrapper trains synchronous pipelines.
