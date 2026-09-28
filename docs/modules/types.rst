Typing helpers
==============

.. py:module:: tpcp.types

The public types in :mod:`tpcp.types` help type custom algorithms, pipelines, datasets, and scoring code.
They describe relationships for static type checkers; importing them does not change how tpcp runs.
Parameter role annotations such as :data:`tpcp.OptimizableParameter` are documented under
:doc:`parameter`.

Type variables
--------------

These variables retain the concrete subclass in a generic function or class. For example, a helper
annotated with :data:`PipelineT` returns the same pipeline type that it accepts::

    from tpcp.types import PipelineT

    def cloned_pipeline(pipeline: PipelineT) -> PipelineT:
        return pipeline.clone()

The :ref:`custom Optuna optimizer example <custom_optuna_optimizer>` uses :data:`PipelineT` and
:data:`DatasetT` to keep its pipeline and dataset types connected.

.. py:data:: BaseTpcpObjectT

   Type variable bound to :class:`tpcp.BaseTpcpObject`. Use it for helpers that preserve the
   concrete type of any tpcp object.

.. py:data:: AlgorithmT

   Type variable bound to :class:`tpcp.Algorithm`. Use it when an algorithm helper returns the
   same concrete algorithm type it receives.

.. py:data:: PipelineT

   Type variable bound to :class:`tpcp.Pipeline`. Use it when a helper or optimizer must preserve
   the concrete pipeline type.

.. py:data:: OptimizablePipelineT

   Type variable bound to :class:`tpcp.OptimizablePipeline`. Use it when the pipeline must support
   :meth:`~tpcp.OptimizablePipeline.self_optimize` and retain its concrete type.

.. py:data:: DatasetT

   Type variable bound to tpcp's dataset base. It covers :class:`tpcp.Dataset` and its subclasses
   and preserves the concrete dataset type in generic code.

Scoring aliases
---------------

The scoring aliases use :data:`PipelineT` and :data:`DatasetT` as their callable input types.
For example, a function that returns one float score can be annotated as::

    from tpcp import Dataset, Pipeline
    from tpcp.types import ScoreFuncSingle

    def score(pipeline: Pipeline, dataset: Dataset) -> float:
        return float(len(dataset))

    scoring: ScoreFuncSingle[Pipeline, Dataset] = score

See :class:`tpcp.validate.Scorer` for score aggregation and named scores.

.. py:data:: ScoreFuncSingle

   A callable that accepts a pipeline and dataset and returns one ``float`` or an
   :class:`~tpcp.validate.Aggregator` wrapping a score.

.. py:data:: ScoreFuncMultiple

   A callable that accepts a pipeline and dataset and returns a dictionary from score names to
   ``float`` values or :class:`~tpcp.validate.Aggregator` instances.

.. py:data:: ScoreFunc

   A callable that accepts a pipeline and dataset and returns either a single score or a
   dictionary of named scores. Use :data:`ScoreFuncSingle` or :data:`ScoreFuncMultiple` when the
   expected form is known.

.. py:data:: AggReturnType

   Return type of :meth:`tpcp.validate.Aggregator.aggregate`: a ``float``, a dictionary of named
   ``float`` values, or :data:`tpcp.NOTHING` when no aggregate score should be emitted.

.. currentmodule:: tpcp

.. py:data:: NOTHING

   Singleton sentinel returned by the aggregator produced by :func:`tpcp.validate.no_agg` to omit
   an aggregate score. It is also used when ``None`` could be a valid value and cannot mean
   "missing".

.. currentmodule:: tpcp.validate

.. py:data:: ScorerTypes

   A :data:`tpcp.types.ScoreFunc` or a :class:`Scorer` instance. Functions such as
   :func:`validate` accept either form as their ``scoring`` argument.
