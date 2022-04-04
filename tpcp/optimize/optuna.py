"""Implementation of methods and classes to wrap the optimization Framework `Optuna`."""
import warnings

try:
    import optuna  # noqa: unused-import
except ImportError as e:
    raise ImportError(
        "To use the tpcp Optuna interface, you first need to install optuna (`pip install optuna`)"
    ) from e

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from optuna import Study, Trial
from optuna.structs import FrozenTrial
from optuna.study.study import ObjectiveFuncType
from typing_extensions import Self

from tpcp import OptimizablePipeline, clone
from tpcp._dataset import DatasetT
from tpcp._optimize import BaseOptimize
from tpcp._pipeline import PipelineT
from tpcp.optimize import Optimize

__all__ = ["CustomOptunaOptimize"]

warnings.warn(
    "The Optuna interface in tpcp is still experimental and we are testing if the workflow makes "
    "sense even for larger projects. "
    "This means, the interface for `CustomOptunaOptimize` will likely change in the future.",
    UserWarning,
)


class CustomOptunaOptimize(BaseOptimize[PipelineT, DatasetT]):
    """Base class for custom Optuna optimizer.

    This provides a relatively simple tpcp compatible interface to Optuna.
    You basically need to subclass this class and implement the `create_objective` method to return the objective
    function you want to optimize.
    The only difference to a normal objective function in Optuna is, that your objective here should expect a
    pipeline and a dataset object as second and third argument (see Example).
    If there are parameters you want to make customizable (e.g. which metric to optimize for), expose them in the
    `__init__` of your subclass.

    Depending on your usecase, your custom optimizers can be single use with a bunch of "hard-coded" logic, or you can
    try to make them more general, by exposing certain configurability.

    Parameters
    ----------
    pipeline
        A tpcp pipeline with some hyper-parameters that should be optimized.
        This can either be a normal pipeline or an optimizable-pipeline.
        This fully depends on your implementation of the `create_objective` method.
    study
        The optuna :class:`~optuna.Study` that should be used for optimization.
    n_trials
        The number of trials.
        If this argument is set to :obj:`None`, there is no limitation on the number of trials.
        In this case you should use :obj:`timeout` instead.
        Because optuna is called internally by this wrapper, you can not setup a study without limits and end it
        using CTRL+C (as suggested by the Optuna docs).
        In this case the entire execution flow would be stopped.
    timeout
        Stop study after the given number of second(s).
        If this argument is set to :obj:`None`, the study is executed without time limitation.
        In this case you should use :obj:`n_trials` to limit the execution.
    return_optimized
        If True, a pipeline object with the overall best parameters is created and re-optimized using all provided data
        as input.
        The optimized pipeline object is stored as `optimized_pipeline_`.
        How the "re-optimization" works depends on the type of pipeline provided.
        If it is a simple pipeline, no specific re-optimization will be perfomed and `optimized_pipeline_` will simply
        be an instance of the pipeline with the best parameters indentified in the search.
        When `pipeline` is a subclass of `OptimizablePipeline`, we attempt to call `pipeline.self_optimize` with the
        entire dataset provided to the `optimize` method.
        The result of this self-optimization will be set as `optimized_pipeline`.
        If this behaviour is undesired, you can overwrite the `return_optimized_pipeline` method in subclass.s
    callbacks
        List of callback functions that are invoked at the end of each trial.
        Each function must accept two parameters with the following types in this order:
        :class:`~optuna.study.Study` and :class:`~optuna.FrozenTrial`.
    show_progress_bar
        Flag to show progress bars or not.
    gc_after_trial
        Run the garbage collerctor after each trial.
        Check the optuna documentation for more detail

    Other Parameters
    ----------------
    dataset
        The dataset instance passed to the optimize method

    Attributes
    ----------
    search_results_
        A dictionary containing all relevant results of the parameter search.
        The format of this dictionary is designed to be directly passed into the `pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters.

        The dictionary contains the following entries:

        score
            The value of the score for a specific trial.
            If a trial was pruned this value is nan.
        param_{parameter_name}
            The value of a respective parameter.
        params
            A dictionary representing all parameters.
        state
            Whether the trial was completed, pruned, or any type of error occured
        user_attrs
        datetime_start
            When the trial was started
        datetime_complete
            When the trial endend
        duration
            The duration of the trial
        user_attrs_...
            User attributes set within the objective function
        system_attrs_...
            System attributes set internally by optuna (usually empty)

        If you need access to further parameter, inspect `self.study_` directly.
    optimized_pipeline_
        An instance of the input pipeline with the best parameter set.
        This is only available if `return_optimized` is not False.
    best_params_
        The parameter combination identified in the study
    best_score_
        The score achieved in the best trial
    best_trial_
        The trial object that resulted in the best results
    study_
        The study object itself.
        This should usually be identical to `self.study`.

    Examples
    --------
    >>> from tpcp.validate import Scorer
    >>> from optuna import create_study
    >>> from optuna import samplers
    >>>
    >>> class MyOptunaOptimizer(CustomOptunaOptimize):
    ...     def create_objective(self):
    ...         def objective(trial: Trial, pipeline: Pipeline, dataset: Dataset):
    ...             trial.suggest_float("my_pipeline_para", 0, 3)
    ...             mean_score = Scorer(lambda dp: pipeline.score(dp))
    ...             return mean_score
    ...         return objective
    >>>
    >>> study = create_study(sampler=samplers.RandomSampler())
    >>> opti = MyOptunaOptimizer(pipeline=MyPipeline(), study=study, n_trials=10)
    >>> opti = opti.optimize(MyDataset())

    Notes
    -----
    As this wrapper attempts to fully encapsule all Optuna calls to make it possible to be run seamlessly in a
    cross-validation (or similar), you can not start multiple optuna optimizations at the same time which is the
    preffered way of multi-processing for optuna.
    In result, you are limited to single-process operations.
    If you want to get "hacky" you can try the approach suggested
    `here <https://github.com/optuna/optuna/issues/2862>`__ to create a study that uses joblib for internal
    multiprocessing.

    """

    pipeline: PipelineT
    study: Study

    return_optimized: bool

    # Optuna Parameters that are directly forwarded to study.optimize
    n_trials: Optional[int]
    timeout: Optional[float]
    callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]]
    gc_after_trial: bool
    show_progress_bar: bool

    optimized_pipeline_: PipelineT
    study_: Study

    def __init__(
        self,
        pipeline: PipelineT,
        study: Study,
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
        return_optimized: bool = True,
    ) -> None:  # noqa: super-init-not-called
        self.pipeline = pipeline
        self.study = study

        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks
        self.gc_after_trial = gc_after_trial
        self.show_progress_bar = show_progress_bar

        self.return_optimized = return_optimized

    @property
    def search_results_(self) -> Dict[str, Sequence[Any]]:
        """Detailed results of the study.

        This basically contains the same information as `self.study_.trials_dataframe()`, with some small modifications:

        - columns starting with "params_" are renamed to "param_"
        - a new column called "params" containing all parameters as dict is added
        - "value" is renamed to score"
        - the score of pruned trials is set to `np.nan`

        These changes are made to make the output comparable to the output of :class:`~tpcp.optimize.GridSearch` and
        :class:`~tpcp.optimize.GridSearchCV`.
        """

        def rename_param_columns(name: str):
            param_start = "params_"
            if name.startswith(param_start):
                return "param_" + name[len(param_start) :]
            return name

        base_df = (
            self.study_.trials_dataframe()
            .drop("number", axis=1)
            .rename(columns={"value": "score"})
            .rename(columns=rename_param_columns)
        )
        base_df["params"] = [t.params for t in self.study_.trials]

        # If a trial is pruned we set the score to nan.
        # This is clearer than showing the last step value.
        # If this is required people should report them as user values in their objective function
        base_df.loc[base_df["state"] == "PRUNED", "score"] = np.nan

        return base_df.to_dict(orient="list")

    @property
    def best_params_(self) -> Dict[str, Any]:
        """Parameters of the best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_params

    @property
    def best_score_(self) -> float:
        """Best score reached in the study."""
        return self.study_.best_value

    @property
    def best_trial_(self) -> FrozenTrial:
        """Best trial in the :class:`~optuna.study.Study`."""
        return self.study_.best_trial

    def optimize(self, dataset: DatasetT, **_: Any) -> Self:
        """Optimize the objective over the dataset and find the best parameter combination.

        This method calls `self.create_objective` to obtain the objective function that should be optimized.

        Parameters
        ----------
        dataset
            The dataset used for optimization.

        """
        if self.timeout is None and self.n_trials is None:
            raise ValueError(
                "You need to set either `timeout` or `n_trials` to a proper value."
                "Otherwise the optimization will not stop und run until infinity."
            )

        self.dataset = dataset

        objective = self._create_objective(self.pipeline, dataset=dataset)
        self.study_ = self._call_optimize(self.study, objective)

        if self.return_optimized:
            self.optimized_pipeline_ = self.return_optimized_pipeline(clone(self.pipeline), dataset, self.study_)
        return self

    def _call_optimize(self, study: Study, objective: ObjectiveFuncType) -> Study:
        """Call the optuna study.

        This is a separate method to make it easy to modify how the study is called.
        """
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=self.callbacks,
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar,
        )
        return study

    def create_objective(self) -> Callable[[Trial, PipelineT, DatasetT], Union[float, Sequence[float]]]:
        """Return the objective function that should be optimized.

        This method should be implemented by a child class and return an objective function that is compatible with
        Optuna.
        However, compared to a normal Optuna objective function, the function should expect a pipeline and a dataset
        object as additional inputs to the optimization Trial object.
        """
        raise NotImplementedError()

    def _create_objective(self, pipeline: PipelineT, dataset: DatasetT) -> ObjectiveFuncType:
        inner_objective = self.create_objective()

        def objective(trial: Trial):
            inner_pipe = clone(pipeline)
            return inner_objective(trial, inner_pipe, dataset)

        return objective

    def return_optimized_pipeline(  # noqa: no-self-use
        self, pipeline: PipelineT, dataset: DatasetT, study: Study
    ) -> PipelineT:
        """Return the pipeline with the best parameters of a study.

        This either just returns the pipeline with the best parameters set, or if the pipeline is a subclass of
        `OptimizablePipeline` it attempts a re-optimization of the pipeline using the provided dataset.

        This functionality is a sensible default, but it is expected to overwrite this method in custom subclasses,
        if specific behaviour is needed.

        Don't call this function on its own! It is only expected to be called internally by optimize.
        """
        # Pipeline that will be passed here is already cloned, so no need to clone again.
        pipeline_with_best_params = pipeline.set_params(**study.best_params)
        if isinstance(pipeline_with_best_params, OptimizablePipeline):
            return Optimize(pipeline_with_best_params).optimize(dataset).optimized_pipeline_
        return pipeline_with_best_params
