Custom Algorithms and Pipelines
===============================

Classes
-------

.. currentmodule:: tpcp

.. autosummary::
   :toctree: generated/algorithm
   :template: class_with_private.rst

    Algorithm
    Pipeline
    OptimizablePipeline


Functions
---------

.. currentmodule:: tpcp

.. autosummary::
   :toctree: generated/algorithm
   :template: function.rst

    make_action_safe
    make_optimize_safe
    get_param_names
    get_action_params
    get_results
    get_config
    set_config
    config_context

The :ref:`action safety example <action_safety>` shows how to check action methods and configure checks in parallel tasks.

The :doc:`typing helpers <types>` include type variables for algorithms and pipelines that retain
the concrete subclass in generic code.

Async actions
-------------

.. autosummary::
   :toctree: generated/aio
   :template: class_with_private.rst

    tpcp.aio.Algorithm
    tpcp.aio.Pipeline
    tpcp.aio.OptimizablePipeline

To compose native async actions, subclass ``Algorithm``,
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
