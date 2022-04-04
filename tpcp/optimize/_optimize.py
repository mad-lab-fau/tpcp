"""Higher level wrapper to run training and parameter optimizations."""
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory
from typing import Any, ContextManager, Dict, Generic, Iterator, List, Optional, Union

import numpy as np
from joblib import Memory, delayed
from numpy.ma import MaskedArray
from scipy.stats import rankdata
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, check_cv
from tqdm.auto import tqdm
from typing_extensions import Self

from tpcp import OptimizablePipeline
from tpcp._algorithm_utils import OPTIMIZE_METHOD_INDICATOR, _check_safe_optimize
from tpcp._base import _get_annotated_fields_of_type
from tpcp._dataset import DatasetT
from tpcp._optimize import BaseOptimize
from tpcp._parameters import Parameter, _ParaTypes
from tpcp._pipeline import OptimizablePipelineT, PipelineT
from tpcp._utils._general import (
    _aggregate_final_results,
    _normalize_score_results,
    _prefix_para_dict,
    _split_hyper_and_pure_parameters,
)
from tpcp._utils._multiprocess import TqdmParallel
from tpcp._utils._score import _ERROR_SCORE_TYPE, _optimize_and_score, _score
from tpcp.exceptions import PotentialUserErrorWarning
from tpcp.validate._scorer import ScorerTypes, ScoreTypeT, _validate_scorer


class DummyOptimize(BaseOptimize[PipelineT, DatasetT], _skip_validation=True):
    """Provide API compatibility for SimplePipelines in optimize wrappers.

    This is a simple dummy Optimizer that will **not** optimize anything, but just provide the correct API so that
    pipelines that do not have the possibility to be optimized can be passed to wrappers like
    :func:`tpcp.validate.cross_validate`.


    Parameters
    ----------
    pipeline
        The pipeline to wrap.
        It will not be optimized in any way, but simply copied to `self.optimized_pipeline_` if `optimize` is called.

    Other Parameters
    ----------------
    dataset
        The dataset used for optimization.
        As no optimization is performed, this will be ignored.

    Attributes
    ----------
    optimized_pipeline_
        The optimized version of the pipeline.
        In case of this class, this is just an unmodified clone of the input pipeline.

    """

    pipeline: Parameter[PipelineT]

    optimized_pipeline_: PipelineT

    def __init__(self, pipeline: PipelineT) -> None:  # noqa: super-init-not-called
        self.pipeline = pipeline

    def optimize(self, dataset: DatasetT, **optimize_params: Any) -> Self:
        """Run the "dummy" optimization.

        Parameters
        ----------
        dataset
            The parameter is ignored, as no real optimization is performed
        optimize_params
            The parameter is ignored, as no real optimization is performed

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.dataset = dataset
        if hasattr(self.pipeline, "self_optimize"):
            warnings.warn(
                "You are using `DummyOptimize` with a pipeline that implements `self_optimize` and, hence, indicates "
                "that the pipeline can be optimized. "
                "`DummyOptimize` does never call this method and skips any optimization steps! "
                "Use `Optimize` if you actually want to optimize your pipeline.",
                PotentialUserErrorWarning,
            )
        self.optimized_pipeline_ = self.pipeline.clone()
        return self


class Optimize(BaseOptimize[OptimizablePipelineT, DatasetT]):
    """Run a generic self-optimization on the pipeline.

    This is a simple wrapper for pipelines that already implement a `self_optimize` method.
    This wrapper can be used to ensure that these algorithms can be optimized with the same interface as other
    optimization methods and can hence be used in methods like :func:`tpcp.validate.cross_validate`.

    Optimize will never modify the original pipeline, but will store a copy of the optimized pipeline as
    `optimized_pipeline_`.

    If `safe_optimize` is True, the wrapper applies the same runtime checks as provided by
    :func:`~tpcp.make_optimize_safe`.

    Parameters
    ----------
    pipeline
        The pipeline to optimize. The pipeline must implement `self_optimize` to optimize its own input parameters.
    safe_optimize
        If True, we add additional checks to make sure the `self_optimize` method of the pipeline is correctly
        implemented.
        See :func:`~tpcp.make_optimize_safe` for more info.

    Other Parameters
    ----------------
    dataset
        The dataset used for optimization.

    Attributes
    ----------
    optimized_pipeline_
        The optimized version of the pipeline.
        That is a copy of the input pipeline with modified params.

    """

    pipeline: Parameter[OptimizablePipelineT]
    safe_optimize: bool

    optimized_pipeline_: OptimizablePipelineT

    def __init__(  # noqa: super-init-not-called
        self, pipeline: OptimizablePipelineT, *, safe_optimize: bool = True
    ) -> None:
        self.pipeline = pipeline
        self.safe_optimize = safe_optimize

    def optimize(self, dataset: DatasetT, **optimize_params: Any) -> Self:
        """Run the self-optimization defined by the pipeline.

        The optimized version of the pipeline is stored as `self.optimized_pipeline_`.

        Parameters
        ----------
        dataset
            An instance of a :class:`~tpcp.dataset.Dataset` containing one or multiple data points that can
            be used for optimization.
            The structure of the data and the available reference information will depend on the dataset.
        optimize_params
            Additional parameter for the optimization process.
            They are forwarded to `pipeline.self_optimize`.

        Returns
        -------
        self
            The class instance with all result attributes populated

        """
        self.dataset = dataset
        if not hasattr(self.pipeline, "self_optimize"):
            raise ValueError(
                "To use `Optimize` with a pipeline, the pipeline needs to implement a `self_optimize` method."
            )
        # We clone just to make sure runs are independent
        pipeline: OptimizablePipeline = self.pipeline.clone()
        if self.safe_optimize is True:
            # We check here, if the pipeline already has the safe decorator and if yes just call it.
            if getattr(pipeline.self_optimize, OPTIMIZE_METHOD_INDICATOR, False) is True:
                optimized_pipeline = pipeline.self_optimize(dataset, **optimize_params)
            else:
                optimized_pipeline = _check_safe_optimize(pipeline, pipeline.self_optimize, dataset, **optimize_params)
        else:
            optimized_pipeline = pipeline.self_optimize(dataset, **optimize_params)
        # We clone again, just to be sure
        self.optimized_pipeline_ = optimized_pipeline.clone()
        return self


class GridSearch(BaseOptimize[PipelineT, DatasetT], Generic[PipelineT, DatasetT, ScoreTypeT]):
    """Perform a grid search over various parameters.

    This scores the pipeline for every combination of data points in the provided dataset and parameter combinations
    in the `parameter_grid`.
    The scores over the entire dataset are then aggregated for each parameter combination.
    By default, this aggregation is a simple average.

    .. note::
        This is different to how grid search works in many other cases:
        Usually, the performance parameter would be calculated on all data points at once.
        Here, each data point represents an entire participant or recording (depending on the dataset).
        Therefore, the pipeline and the scoring method are expected to provide a result/score per data point
        in the dataset.
        Note that it is still open to your interpretation what you consider a "data point" in the context of your
        analysis. The `run` method of the pipeline can still process multiple data points, e.g., gait tests, in a loop
        and generate a single output if you consider a single participant *one data point*.

    Parameters
    ----------
    pipeline
        The pipeline object to optimize
    parameter_grid
        A sklearn parameter grid to define the search space.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the pipeline is used instead.

        Note that if scoring returns a dictionary, `return_optimized` must be set to the name of the score that
        should be used for ranking.
    n_jobs
        The number of processes that should be used to parallelize the search.
        `None` means 1 while -1 means as many as logical processing cores.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~sklearn.model_selection.GridSearchCV`
    return_optimized
        If True, a pipeline object with the overall best params is created and stored as `optimized_pipeline_`.
        If `scoring` returns a dictionary of score values, this must be a `str` corresponding to the name of the
        score that should be used to rank the results.
        If False, the respective result attributes will not be populated.
        If multiple parameter combinations have the same score, the one tested first will be used.
        Otherwise, higher values are always considered better.
    error_score
        Value to assign to the score if an error occurs during scoring.
        If set to ‘raise’, the error is raised.
        If a numeric value is given, a Warning is raised.
    progress_bar
        True/False to enable/disable a tqdm progress bar.

    Other Parameters
    ----------------
    dataset
        The dataset instance passed to the optimize method

    Attributes
    ----------
    gs_results_
        A dictionary summarizing all results of the gridsearch.
        The format of this dictionary is designed to be directly passed into the :class:`~pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters

        The dictionary contains the following entries:

        param_*
            The value of a respective parameter
        params
            A dictionary representing all parameters
        score / {scorer-name}
            The aggregated value of a score over all data-points.
            If a single score is used for scoring, then the generic name "score" is used.
            Otherwise, multiple columns with the name of the respective scorer exist
        rank_score / rank_{scorer-name}
            A sorting for each score from the highest to the lowest value
        single_score / single_{scorer-name}
            The individual scores per data point for each parameter combination.
            This is a list of values with the `len(dataset)`.
        data_labels
            A list of data labels in the order the single score values are provided.
            These can be used to associate the `single_score` values with a certain data point.
    optimized_pipeline_
        An instance of the input pipeline with the best parameter set.
        This is only available if `return_optimized` is not False.
    best_params_
        The parameter dict that resulted in the best result.
        This is only available if `return_optimized` is not False.
    best_index_
        The index of the result row in the output.
        This is only available if `return_optimized` is not False.
    best_score_
        The score of the best result.
        In a multimetric case, only the value of the scorer specified by `return_optimized` is provided.
        This is only available if `return_optimized` is not False.
    multimetric_
        If the scorer returned multiple scores

    """

    parameter_grid: ParameterGrid
    scoring: ScorerTypes[PipelineT, DatasetT, ScoreTypeT]
    n_jobs: Optional[int]
    return_optimized: Union[bool, str]
    pre_dispatch: Union[int, str]
    error_score: _ERROR_SCORE_TYPE
    progress_bar: bool

    gs_results_: Dict[str, Any]
    best_params_: Dict[str, Any]
    best_index_: int
    best_score_: float
    multimetric_: bool

    def __init__(  # noqa: super-init-not-called
        self,
        pipeline: PipelineT,
        parameter_grid: ParameterGrid,
        *,
        scoring: ScorerTypes[PipelineT, DatasetT, ScoreTypeT] = None,
        n_jobs: Optional[int] = None,
        return_optimized: Union[bool, str] = True,
        pre_dispatch: Union[int, str] = "n_jobs",
        error_score: _ERROR_SCORE_TYPE = np.nan,
        progress_bar: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.return_optimized = return_optimized
        self.error_score = error_score
        self.progress_bar = progress_bar

    def optimize(self, dataset: DatasetT, **_: Any) -> Self:
        """Run the grid search over the dataset and find the best parameter combination.

        Parameters
        ----------
        dataset
            The dataset used for optimization.

        """
        self.dataset = dataset
        scoring = _validate_scorer(self.scoring, self.pipeline)

        # We use a similar structure as sklearn's GridSearchCV here, but instead of calling something equivalent to
        # `fit_score`, we call `score`, which just applies and scores the pipeline on the entirety of our dataset as
        # we do not need a "train" step.
        # Our main loop just loops over all parameter combinations and the `_score` function then applies the parameter
        # combination to the pipeline and scores the resulting pipeline on the dataset, by passing the entire dataset
        # and the pipeline to the scorer.
        # Looping over the individual data points in the dataset and aggregating the scores is handled by the scorer
        # itself.
        # If not explicitly changed the scorer is an instance of `Scorer` that wraps the actual `scoring`
        # function provided by the user.
        pbar: Optional[tqdm] = None
        if self.progress_bar:
            pbar = tqdm(total=len(self.parameter_grid), desc="Parameter Combinations")
        parallel = TqdmParallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, pbar=pbar)
        with parallel:
            # Evaluate each parameter combination
            results = parallel(
                delayed(_score)(
                    self.pipeline.clone(),
                    dataset,
                    scoring,
                    paras,
                    return_parameters=True,
                    return_data_labels=True,
                    return_times=True,
                    error_score=self.error_score,
                )
                for paras in self.parameter_grid
            )
        assert results is not None  # For the typechecker
        # We check here if all results are dicts. We only check the dtype of the first value, as the scorer should
        # have handled issues with non-uniform cases already.
        first_test_score = results[0]["scores"]
        self.multimetric_ = isinstance(first_test_score, dict)
        _validate_return_optimized(self.return_optimized, self.multimetric_, first_test_score)

        results = self._format_results(
            list(self.parameter_grid),
            results,
        )

        if self.return_optimized:
            return_optimized = "score"
            if self.multimetric_ and isinstance(self.return_optimized, str):
                return_optimized = self.return_optimized
            self.best_index_ = results[f"rank_{return_optimized}"].argmin()
            self.best_score_ = results[return_optimized][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
            # We clone twice, in case one of the params was itself an algorithm.
            self.optimized_pipeline_ = self.pipeline.clone().set_params(**self.best_params_).clone()

        self.gs_results_ = results

        return self

    def _format_results(self, candidate_params, out):  # noqa: no-self-use
        """Format the final result dict.

        This function is adapted based on sklearn's `BaseSearchCV`
        """
        n_candidates = len(candidate_params)
        out = _aggregate_final_results(out)

        results = {}

        scores_dict = _normalize_score_results(out["scores"])
        single_scores_dict = _normalize_score_results(out["single_scores"])
        for c, v in scores_dict.items():
            results[c] = v
            results[f"rank_{c}"] = np.asarray(rankdata(-v, method="min"), dtype=np.int32)
            results[f"single_{c}"] = single_scores_dict[c]

        results["data_labels"] = out["data_labels"]
        results["score_time"] = out["score_time"]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results[f"param_{name}"][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        return results


class GridSearchCV(BaseOptimize[OptimizablePipelineT, DatasetT], Generic[OptimizablePipelineT, DatasetT, ScoreTypeT]):
    """Exhaustive (hyper)parameter search using a cross validation based score to optimize pipeline parameters.

    This class follows as much as possible the interface of :func:`~sklearn.model_selection.GridSearchCV`.
    If the `tpcp` documentation is missing some information, the respective documentation of `sklearn` might be helpful.

    Compared to the `sklearn` implementation this method uses a couple of `tpcp`-specific optimizations and
    quality-of-life improvements.

    Parameters
    ----------
    pipeline
        A tpcp pipeline implementing `self_optimize`.
    parameter_grid
        A sklearn parameter grid to define the search space for the grid search.
    scoring
        A callable that can score a single data point given a pipeline.
        This function should return either a single score or a dictionary of scores.
        If scoring is `None` the default `score` method of the pipeline is used instead.

        .. note:: If scoring returns a dictionary, `return_optimized` must be set to the name of the score that
                  should be used for ranking.
    return_optimized
        If True, a pipeline object with the overall best parameters is created and re-optimized using all provided data
        as input.
        The optimized pipeline object is stored as `optimized_pipeline_`.
        If `scoring` returns a dictionary of score values, this must be a str corresponding to the name of the
        score that should be used to rank the results.
        If False, the respective result attributes will not be populated.
        If multiple parameter combinations have the same mean score over all CV folds, the one tested first will be
        used.
        Otherwise, higher mean values are always considered better.
    cv
        An integer specifying the number of folds in a K-Fold cross validation or a valid cross validation helper.
        The default (`None`) will result in a 5-fold cross validation.
        For further inputs check the `sklearn`
        `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_.
    pure_parameters
        .. warning::
            Do not use this option unless you fully understand it!

        A list of parameter names (named in the `parameter_grid`) that do not affect training aka are not
        hyperparameters.
        This information can be used for massive performance improvements, as the training does not need to be
        repeated if one of these parameters changes.
        However, setting it incorrectly can lead detect errors that are very hard to detect in the final results.

        Instead of passing a list of names, you can also just set the value to `True`.
        In this case all parameters of the provided pipeline that are marked as :func:`~tpcp.pure_parameter` are used.
        Note that pure parameters of nested objects are not considered, but only top-level attributes.
        If you need to mark nested parameters as pure, use the first method and pass the names (with `__`) as part of
        the list of names.

        For more information on this approach see the :func:`evaluation guide <algorithm_evaluation>`.
    return_train_score
        If True the performance on the train score is returned in addition to the test score performance.
        Note, that this increases the runtime.
        If `True`, the fields `train_score`, and `train_score_single` are available in the results.
    verbose
        Control the verbosity of information printed during the optimization (larger number -> higher verbosity).
        At the moment this will only affect the caching done, when `pure_parameter_names` are provided.
    n_jobs
        The number of parallel jobs.
        The default (`None`) means 1 job at the time, hence, no parallel computing.
        -1 means as many as logical processing cores.
        One job is created per cv + para combi combination.
    pre_dispatch
        The number of jobs that should be pre dispatched.
        For an explanation see the documentation of :class:`~sklearn.model_selection.GridSearchCV`
    error_score
        Value to assign to the score if an error occurs during scoring.
        If set to ‘raise’, the error is raised.
        If a numeric value is given, a Warning is raised.
    progress_bar
        True/False to enable/disable a tqdm progressbar.
    safe_optimize
        If True, we add additional checks to make sure the `self_optimize` method of the pipeline is correctly
        implemented.
        See :func:`~tpcp.make_optimize_safe` for more info.

    Other Parameters
    ----------------
    dataset
        The dataset instance passed to the optimize method

    Attributes
    ----------
    cv_results_
        A dictionary summarizing all results of the gridsearch.
        The format of this dictionary is designed to be directly passed into the `pd.DataFrame` constructor.
        Each column then represents the result for one set of parameters.

        The dictionary contains the following entries:

        param_{parameter_name}
            The value of a respective parameter.
        params
            A dictionary representing all parameters.
        mean_test_score / mean_test_{scorer_name}
            The average test score over all folds.
            If a single score is used for scoring, then the generic name "score" is used.
            Otherwise, multiple columns with the name of the respective scorer exist.
        std_test_score / std_test_{scorer_name}
            The std of the test scores over all folds.
        rank_test_score / rank_{scorer_name}
            The rank of the mean test score assuming higher values are better.
        split{n}_test_score / split{n}_test_{scorer_name}
            The performance on the test set in fold n.
        split{n}_test_single_score / split{n}_test_single_{scorer_name}
            The performance in fold n on every single data point in the test set.
        split{n}_test_data_labels
            The ids of the data points used in the test set of fold n.
        mean_train_score / mean_train_{scorer_name}
            The average train score over all folds.
        std_train_score / std_train_{scorer_name}
            The std of the train scores over all folds.
        split{n}_train_score / split{n}_train_{scorer_name}
            The performance on the train set in fold n.
        rank_train_score / rank_{scorer_name}
            The rank of the mean train score assuming higher values are better.
        split{n}_train_single_score / split{n}_train_single_{scorer_name}
            The performance in fold n on every single datapoint in the train set.
        split{n}_train_data_labels
            The ids of the data points used in the train set of fold n.
        mean_{optimize/score}_time
            Average time over all folds spent for optimization and scoring, respectively.
        std_{optimize/score}_time
            Standard deviation of the optimize/score times over all folds.

    optimized_pipeline_
        An instance of the input pipeline with the best parameter set.
        This is only available if `return_optimized` is not False.
    best_params_
        The parameter dict that resulted in the best result.
        This is only available if `return_optimized` is not False.
    best_index_
        The index of the result row in the output.
        This is only available if `return_optimized` is not False.
    best_score_
        The score of the best result.
        In a multimetric case, only the value of the scorer specified by `return_optimized` is provided.
        This is only available if `return_optimized` is not False.
    multimetric_
        If the scorer returned multiple scores
    final_optimize_time_
        Time spent to perform the final optimization on all data.
        This is only available if `return_optimized` is not False.

    """

    pipeline: OptimizablePipelineT
    parameter_grid: ParameterGrid
    scoring: ScorerTypes[OptimizablePipelineT, DatasetT, ScoreTypeT]
    return_optimized: Union[bool, str]
    cv: Optional[Union[int, BaseCrossValidator, Iterator]]
    pure_parameters: Union[bool, List[str]]
    return_train_score: bool
    verbose: int
    n_jobs: Optional[int]
    pre_dispatch: Union[int, str]
    error_score: _ERROR_SCORE_TYPE
    progress_bar: bool
    safe_optimize: bool

    cv_results_: Dict[str, Any]
    best_params_: Dict[str, Any]
    best_index_: int
    best_score_: float
    multimetric_: bool
    final_optimize_time_: float

    def __init__(  # noqa: super-init-not-called
        self,
        pipeline: OptimizablePipelineT,
        parameter_grid: ParameterGrid,
        *,
        scoring: ScorerTypes[OptimizablePipelineT, DatasetT, ScoreTypeT] = None,
        return_optimized: Union[bool, str] = True,
        cv: Optional[Union[int, BaseCrossValidator, Iterator]] = None,
        pure_parameters: Union[bool, List[str]] = False,
        return_train_score: bool = False,
        verbose: int = 0,
        n_jobs: Optional[int] = None,
        pre_dispatch: Union[int, str] = "n_jobs",
        error_score: _ERROR_SCORE_TYPE = np.nan,
        progress_bar: bool = True,
        safe_optimize: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.scoring = scoring
        self.return_optimized = return_optimized
        self.cv = cv
        self.pure_parameters = pure_parameters
        self.return_train_score = return_train_score
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.progress_bar = progress_bar
        self.safe_optimize = safe_optimize

    def optimize(self, dataset: DatasetT, *, groups=None, **optimize_params) -> Self:  # noqa: arguments-differ
        self.dataset = dataset
        scoring = _validate_scorer(self.scoring, self.pipeline)

        cv_checked: BaseCrossValidator = check_cv(self.cv, None, classifier=True)
        n_splits = cv_checked.get_n_splits(dataset, groups=groups)

        # We need to wrap our pipeline for a consistent interface.
        # In the future we might be able to allow objects with optimizer Interface as input directly.
        optimizer = Optimize(self.pipeline, safe_optimize=self.safe_optimize)

        # For each para combi, we separate the pure parameters (parameters that do not affect the optimization) and
        # the hyperparameters.
        # This allows for massive caching optimizations in the `_optimize_and_score`.
        pure_parameters: Optional[List[str]]
        if self.pure_parameters is False:
            pure_parameters = None
        elif self.pure_parameters is True:
            pure_parameters = _get_annotated_fields_of_type(self.pipeline, _ParaTypes.PURE)
        elif isinstance(self.pure_parameters, list):
            pure_parameters = self.pure_parameters
        else:
            raise ValueError(
                "`self.pure_parameters` must either be a List of field names (nested are allowed) or " "True/False."
            )

        parameters = list(self.parameter_grid)
        split_parameters = _split_hyper_and_pure_parameters(parameters, pure_parameters)
        parameter_prefix = "pipeline__"
        combinations = list(product(enumerate(split_parameters), enumerate(cv_checked.split(dataset, groups=groups))))

        pbar: Optional[tqdm] = None
        if self.progress_bar:
            pbar = tqdm(total=len(combinations), desc="Split-Para Combos")

        # To enable the pure parameter performance improvement, we need to create a joblib cache in a temp dir that
        # is deleted after the run.
        # We only allow a temporary cache here, because the method that is cached internally is generic and the cache
        # might not be correctly invalidated, if GridSearchCv is called with a different pipeline or when the
        # pipeline itself is modified.
        tmp_dir_context: Union[ContextManager[None], TemporaryDirectory] = nullcontext()
        if pure_parameters:
            tmp_dir_context = TemporaryDirectory("joblib_tpcp_cache")
        with tmp_dir_context as cachedir:
            tmp_cache = Memory(cachedir, verbose=self.verbose) if cachedir else None
            parallel = TqdmParallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, pbar=pbar)
            # We use a similar structure to sklearn's GridSearchCV here (see GridSearch for more info).
            with parallel:
                # Evaluate each parameter combination
                out = parallel(
                    delayed(_optimize_and_score)(
                        optimizer.clone(),
                        dataset,
                        scoring,
                        train,
                        test,
                        optimize_params=optimize_params,
                        hyperparameters=_prefix_para_dict(hyper_paras, parameter_prefix),
                        pure_parameters=_prefix_para_dict(pure_paras, parameter_prefix),
                        return_train_score=self.return_train_score,
                        return_parameters=False,
                        return_data_labels=True,
                        return_times=True,
                        error_score=self.error_score,
                        memory=tmp_cache,
                    )
                    for (cand_idx, (hyper_paras, pure_paras)), (split_idx, (train, test)) in combinations
                )
        assert out is not None  # For the type checker
        results = self._format_results(parameters, n_splits, out)
        self.cv_results_ = results

        first_test_score = out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)
        _validate_return_optimized(self.return_optimized, self.multimetric_, first_test_score)
        if self.return_optimized:
            return_optimized = "score"
            if self.multimetric_ and isinstance(self.return_optimized, str):
                return_optimized = self.return_optimized
            self.best_index_ = results[f"rank_test_{return_optimized}"].argmin()
            self.best_score_ = results[f"mean_test_{return_optimized}"][self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
            # We clone twice, in case one of the params was itself an algorithm.
            best_optimizer = Optimize(self.pipeline.clone().set_params(**self.best_params_).clone())
            final_optimize_start_time = time.time()
            optimize_params_clean = optimize_params or {}
            self.optimized_pipeline_ = best_optimizer.optimize(dataset, **optimize_params_clean).optimized_pipeline_
            self.final_optimize_time_ = final_optimize_start_time - time.time()

        return self

    def _format_results(self, candidate_params, n_splits, out, more_results=None):  # noqa: MC0001
        """Format the final result dict.

        This function is adapted based on sklearn's `BaseSearchCV`.
        """
        n_candidates = len(candidate_params)
        out = _aggregate_final_results(out)

        results = dict(more_results or {})

        def _store_non_numeric(key_name: str, array):
            """Store non-numeric scores/times to the cv_results_."""
            # We avoid performing any sort of conversion into numpy arrays as this might modify the dtypes.
            # Instead we use list comprehension to do the reshaping.
            # The result is a list of lists and not a numpy array.
            #
            # Note that the results were produced by iterating first by splits and then by parameters.
            # We do the same here, but directly transpose the results, as we need to access the data per parameters
            # afterwards.
            iterable_array = iter(array)
            array = [[next(iterable_array) for _ in range(n_splits)] for _ in range(n_candidates)]
            # "Transpose" the array
            array = map(list, zip(*array))

            for split_idx, split in enumerate(array):
                # Uses closure to alter the results
                results[f"split{split_idx}_{key_name}"] = split

        def _store(key_name: str, array, weights=None, splits=False, rank=False):
            """Store numeric scores/times to the cv_results_."""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)

            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results[f"split{split_idx}_{key_name}"] = array[:, split_idx]
            array_means = np.average(array, axis=1, weights=weights)
            results[f"mean_{key_name}"] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(~np.isfinite(array_means)):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores are non-finite: {array_means}",
                    category=UserWarning,
                )
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results[f"std_{key_name}"] = array_stds

            if rank:
                results[f"rank_{key_name}"] = np.asarray(rankdata(-array_means, method="min"), dtype=np.int32)

        _store("optimize_time", out["optimize_time"])
        _store("score_time", out["score_time"])
        _store_non_numeric("test_data_labels", out["test_data_labels"])
        _store_non_numeric("train_data_labels", out["train_data_labels"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results: Dict = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_{name}"` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results[f"param_{name}"][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])
        test_single_scores_dict = _normalize_score_results(out["test_single_scores"])

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store(f"test_{scorer_name}", test_scores_dict[scorer_name], splits=True, rank=True, weights=None)
            _store_non_numeric(f"test_single_{scorer_name}", test_single_scores_dict[scorer_name])
            if self.return_train_score:
                train_scores_dict = _normalize_score_results(out["train_scores"])
                train_single_scores_dict = _normalize_score_results(out["train_single_scores"])

                _store(f"train_{scorer_name}", train_scores_dict[scorer_name], splits=True)
                _store_non_numeric(f"train_single_{scorer_name}", train_single_scores_dict[scorer_name])

        return results


def _validate_return_optimized(return_optimized, multi_metric, results) -> None:
    """Check if `return_optimize` fits to the multimetric output of the scorer."""
    if multi_metric is True:
        # In a multimetric case, return_optimized must either be False or a string
        if return_optimized and (not isinstance(return_optimized, str) or return_optimized not in results):
            raise ValueError(
                "If multi-metric scoring is used, `return_optimized` must be a str specifying the score that "
                "should be used to select the best result."
            )
    else:
        if isinstance(return_optimized, str):
            warnings.warn(
                "You set `return_optimized` to the name of a scorer, but the provided scorer only produces a "
                "single score. `return_optimized` is set to True."
            )
