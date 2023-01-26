"""Implementation of methods and classes to wrap the optimization Framework `Optuna`."""
import multiprocessing

try:
    import optuna
except ImportError as e:
    raise ImportError(
        "To use the tpcp Optuna interface, you first need to install optuna (`pip install optuna`)"
    ) from e

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

import joblib
import numpy as np
from optuna import Study, Trial
from optuna.study.study import ObjectiveFuncType
from optuna.trial import FrozenTrial
from typing_extensions import Self

from tpcp import OptimizablePipeline, clone
from tpcp._dataset import DatasetT
from tpcp._optimize import BaseOptimize
from tpcp._pipeline import PipelineT
from tpcp.optimize import Optimize

__all__ = ["CustomOptunaOptimize", "CustomOptunaOptimizeT"]

if multiprocessing.parent_process() is None:
    # We want to avoid spamming the user with warnings if they are running multiple processes
    warnings.warn(
        "The Optuna interface in tpcp is still experimental and we are testing if the workflow makes "
        "sense even for larger projects. "
        "This means, the interface for `CustomOptunaOptimize` will likely change in the future.",
        UserWarning,
    )


CustomOptunaOptimizeT = TypeVar("CustomOptunaOptimizeT", bound="_CustomOptunaOptimize")


def _split_trials(n_trials, n_jobs):
    n_per_job, remaining = divmod(n_trials, n_jobs)
    for _ in range(n_jobs):
        yield n_per_job + (1 if remaining > 0 else 0)
        remaining -= 1


class _CustomOptunaOptimize(BaseOptimize[PipelineT, DatasetT]):
    pipeline: PipelineT
    create_study: Callable[[], Study]

    return_optimized: bool

    # Optuna Parameters that are directly forwarded to study.optimize
    n_trials: Optional[int]
    timeout: Optional[float]
    callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]]
    gc_after_trial: bool
    show_progress_bar: bool
    n_jobs: int

    optimized_pipeline_: PipelineT
    study_: Study

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
        self.study_ = self.create_study()

        if not isinstance(self.study_._storage, optuna.storages.InMemoryStorage):
            warnings.warn(
                "You are using a persistent storage for the study. "
                "This means, that the study will be saved and likely be available after this script "
                "terminates. "
                "To avoid issues and make sure that runs of your optimizations are reproducible and "
                "independent, make sure to cleanup the persistent storage once the results are not needed "
                "anymore.\n"
                "You can use use `optuna.delete_study(study_name=opti_instance.study_.study_name, "
                "storage=opti_instance.study_._storage)`. "
                "Note that all result object that depend on the study are not available anymore after deletion."
            )

        if self.n_jobs == 1:
            self._call_optimize(self.study_, objective)
        else:
            if isinstance(self.study_._storage, optuna.storages.InMemoryStorage):
                raise ValueError(
                    "You are using the InMemoryStorage with n_jobs > 1. "
                    "This will lead to problems as the storage is not persistent and thread safe. "
                    "Use a persistent database based storage instead."
                )
            if self.timeout is not None:
                raise ValueError(
                    "You are using timeout with n_jobs > 1. "
                    "This parameter does not make sense in this case. "
                    "You `n_trials` instead to limit for how long the optimization should run."
                )

            if self.show_progress_bar:
                warnings.warn(
                    "You are using a progress bar with n_jobs > 1. "
                    "This might lead to strange behaviour, as each process will launch its own process bar with "
                    "n_trials/n_jobs steps."
                )

            # This solution is based on the solution proposed here:
            # https://github.com/optuna/optuna/issues/2862
            def _multi_process_call_optimize(n_trials: int):
                study = optuna.load_study(
                    study_name=self.study_.study_name,
                    storage=self.study_._storage,
                    sampler=self.study_.sampler,
                    pruner=self.study_.pruner,
                )
                self._call_optimize_multi_process(study, objective, n_trials)

            parallel = joblib.Parallel(self.n_jobs)
            parallel(
                joblib.delayed(_multi_process_call_optimize)(n_trials=n_trials_i)
                for n_trials_i in _split_trials(self.n_trials, self.n_jobs)
            )

        if self.return_optimized:
            self.optimized_pipeline_ = self.return_optimized_pipeline(clone(self.pipeline), dataset, self.study_)
        return self

    def _call_optimize(self, study: Study, objective: ObjectiveFuncType):
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

    def _call_optimize_multi_process(self, study: Study, objective: ObjectiveFuncType, n_trials: int):
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None,  # This does not work with multiprocessing
            callbacks=self.callbacks,
            gc_after_trial=self.gc_after_trial,
            show_progress_bar=self.show_progress_bar,  # This is a little strange. We warn about it above
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

    def return_optimized_pipeline(self, pipeline: PipelineT, dataset: DatasetT, study: Study) -> PipelineT:
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


class CustomOptunaOptimize(_CustomOptunaOptimize[PipelineT, DatasetT]):
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
    create_study
        A callable that returns an optuna study instance to be used for the optimization.
        It will be called as part of the `optimize` method without parameters.
        The resulting study object can be accessed via `self.study_` after the optimization is finished.
        Creating the study is handled via a callable, instead of providing the study object itself, to make it
        possible to create individual studies, when CustomOptuna optimize is called by an external wrapper
        (i.e. `cross_validate`).
    n_trials
        The number of trials.
        If this argument is set to `None`, there is no limitation on the number of trials.
        In this case you should use `timeout` instead.
        Because optuna is called internally by this wrapper, you can not set up a study without limits and end it
        using CTRL+C (as suggested by the Optuna docs).
        In this case the entire execution flow would be stopped.
    timeout
        Stop study after the given number of second(s).
        If this argument is set to `None`, the study is executed without time limitation.
        In this case you should use `n_trials` to limit the execution.
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
        :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.
    n_jobs
        Number of parallel jobs to use (default = 1 -> single process, -1 -> all available cores).
        This uses joblib with the multiprocessing backend to parallelize the optimization.
        If this is set to -1, all available cores are used.

        .. warning:: Read the notes on multiprocessing below before using this feature.

    show_progress_bar
        Flag to show progress bars or not.
    gc_after_trial
        Run the garbage collector after each trial.
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
    ...             mean_score, _ = Scorer(lambda pipe, dp: pipe.score(dp))(pipeline, dataset)
    ...             return mean_score
    ...         return objective
    >>>
    >>> study = create_study(sampler=samplers.RandomSampler())
    >>> opti = MyOptunaOptimizer(pipeline=MyPipeline(), study=study, n_trials=10)
    >>> opti = opti.optimize(MyDataset())

    Notes
    -----
    Multiprocessing
    ***************
    This class provides a relatively hacky implementation of multiprocessing.
    The implementation is based on the suggestions made here: https://github.com/optuna/optuna/issues/2862
    However, it depends on internal optuna APIs and might break in future versions.

    To use multiprocessing, the provided `create_study` function must return a study with a persistent backend (
    i.e. not the default InMemoryStorage), that can be written to by multiple processes.
    To make sure that your individual runs are independent and you don't leave behind any files, make sure you clean
    up your study after each run. You can use `optuna.delete_study(study_name=opti_instance.study_.study_name,
    storage=opti_instance.study_._storage)` for this.

    From the implementation perspective, we split the number of trials into `n_jobs` chunks and then spawn one study
    per job.
    This study is a copy of the study from the main process and hence, points to the same database.
    Each process will then complete its chunk of trials and then terminate.
    This is a relatively naive implementation, but it avoids the overhead of spawning a new process for each trial.
    If this is always the best idea, is unclear.

    One downside of using multiprocessing is, that your runs will not be reproducible, as the order of the trials is not
    guraranteed and depends on when the individual processes finish.
    This can lead to different suggested parameters when non-trivial samplers are used.
    Note that this is not a specific problem of our implementation, but a general problem of using multiprocessing with
    optuna.

    Further, the use of `show_progress_bar` is not recommended when using multiprocessing, as one progress bar per
    process is
    created and the output is not very readable.
    It might still be helpful to see that something is happening.

    .. note:: Using SQLite as backend is known to cause issues with multiprocessing, when the database is
              stored on a network drive (e.g. as typically done on a cluster).
              On most clusters, you should use the local storage of your node for the database or use a
              different backend (e.g. Redis, MySQL), if multiple nodes need to access the database at once.

    """

    def __init__(
        self,
        pipeline: PipelineT,
        create_study: Callable[[], Study],
        *,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None,
        gc_after_trial: bool = False,
        n_jobs: int = 1,
        show_progress_bar: bool = False,
        return_optimized: bool = True,
    ):
        self.pipeline = pipeline
        self.create_study = create_study

        self.n_trials = n_trials
        self.timeout = timeout
        self.callbacks = callbacks
        self.gc_after_trial = gc_after_trial
        self.show_progress_bar = show_progress_bar
        self.n_jobs = n_jobs
        self.return_optimized = return_optimized

    @staticmethod
    def as_dataclass():
        """Return a version of the Dataset class that can be subclassed using dataclasses."""
        import dataclasses  # pylint: disable=import-outside-toplevel

        @dataclasses.dataclass(eq=False, repr=False, order=False)
        class CustomOptunaOptimizeDc(_CustomOptunaOptimize[PipelineT, DatasetT]):
            """Dataclass version of CustomOptunaOptimize."""

            pipeline: PipelineT
            create_study: Callable[[], Study]

            # Optuna Parameters that are directly forwarded to study.optimize
            n_trials: Optional[int] = None
            timeout: Optional[float] = None
            callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None
            gc_after_trial: bool = False
            show_progress_bar: bool = False

            n_jobs: int = 1
            return_optimized: bool = True

            optimized_pipeline_: PipelineT = dataclasses.field(init=False, repr=False)
            study_: Study = dataclasses.field(init=False, repr=False)

        return CustomOptunaOptimizeDc

    @staticmethod
    def as_attrs():
        """Return a version of the Dataset class that can be subclassed using `attrs` defined classes.

        Note, this requires `attrs` to be installed!
        """
        from attrs import define, field  # pylint: disable=import-outside-toplevel

        @define(eq=False, repr=False, order=False, kw_only=True, slots=False)
        class CustomOptunaOptimizeAt(_CustomOptunaOptimize[PipelineT, DatasetT]):
            """Attrs version of CustomOptunaOptimize."""

            pipeline: PipelineT
            create_study: Callable[[], Study]

            # Optuna Parameters that are directly forwarded to study.optimize
            n_trials: Optional[int] = None
            timeout: Optional[float] = None
            callbacks: Optional[List[Callable[[Study, FrozenTrial], None]]] = None
            gc_after_trial: bool = False
            show_progress_bar: bool = False

            n_jobs: int = (1,)
            return_optimized: bool = True

            optimized_pipeline_: PipelineT = field(init=False, repr=False)
            study_: Study = field(init=False, repr=False)

        return CustomOptunaOptimizeAt
