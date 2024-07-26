"""Helper to score pipelines."""
from __future__ import annotations

import traceback
from collections import defaultdict
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
from joblib import Parallel
from tqdm.auto import tqdm
from typing_extensions import Protocol, Self

from tpcp import NOTHING
from tpcp._base import BaseTpcpObject, _Nothing, cf
from tpcp._dataset import Dataset, DatasetT
from tpcp._hash import custom_hash
from tpcp._pipeline import Pipeline, PipelineT
from tpcp._utils._general import _passthrough
from tpcp.exceptions import ScorerFailedError, ValidationError
from tpcp.parallel import delayed

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")
AggReturnType = Union[float, dict[str, float], _Nothing]


SingleScoreTypeT = Union[T, "Aggregator[Any]"]
MultiScoreTypeT = dict[str, SingleScoreTypeT[T]]
ScoreTypeT = Union[SingleScoreTypeT[T], MultiScoreTypeT[T]]

ScoreFuncSingle = Callable[[PipelineT, DatasetT], SingleScoreTypeT[T]]
ScoreFuncMultiple = Callable[[PipelineT, DatasetT], MultiScoreTypeT[T]]
ScoreFunc = Callable[[PipelineT, DatasetT], ScoreTypeT[T]]


class ScoreCallback(Protocol[PipelineT, DatasetT, T]):
    """Callback signature for scorer callbacks."""

    def __call__(
        self,
        *,
        step: int,
        scores: tuple[ScoreTypeT[T], ...],
        scorer: Scorer[PipelineT, DatasetT, T],
        pipeline: PipelineT,
        dataset: DatasetT,
    ) -> None:
        ...


class Aggregator(BaseTpcpObject, Generic[T]):
    """Base class for aggregators.

    You can subclass this class to create your own aggregators.
    The only thing you should change, is to overwrite the `aggregate` method.
    Everything else should not be modified.

    Custom aggregators can then be used to wrap return values of score functions or they can be passed as
    `default_aggregator` to the :class:`~tpcp.validate.Scorer` class.
    """

    _value: T

    return_raw_scores: bool

    def __init__(self, *, return_raw_scores: bool = True) -> None:
        self.return_raw_scores = return_raw_scores

    def __repr__(self) -> str:
        """Show the representation of the object."""
        return f"{self!r}({self._value!r})"

    def __call__(self, value: T) -> Self:
        """Set the value of the aggregator.

        This will return a clone of itself with the value set.

        .. warning:: Cloning the object again will make it loose the value.
        """
        new_self = self.clone()
        new_self._value = value
        return new_self

    def aggregate(self, /, values: Sequence[T], datapoints: Sequence[Dataset]) -> AggReturnType:
        """Aggregate the values."""
        raise NotImplementedError()

    def get_value(self) -> T:
        """Return the value wrapped by aggregator."""
        if not hasattr(self, "_value"):
            raise AttributeError(
                "Aggregator has no value set yet. "
                "When using a configurable aggregator, first create an instance and then set the value, by calling the instance. "
                "E.g. `my_agg = MyAggregator(); my_agg(42)`."
            )
        return self._value

    def _assert_is_all_valid(self, values: Sequence[Any], _key_name: str):
        """Check if all scoring values are consistently of the same type.

        This methods is called on the first aggregator instance acountered of a scoring value.

        It's role is to check, if all other values are of the same type (aka the same class and same config) as the
        first one.
        """
        encountered_types = {type(v) for v in values}
        if len(encountered_types) > 1 or encountered_types.pop() is not type(self):
            raise ValidationError(
                f"Encountered multiple types of aggregators for the same scoring key. "
                f"Based on the first value encountered for the scoring value {_key_name}, we expected all values to be "
                f"to be wrapped in an aggregator of type {type(self)}. "
                f"However, we encountered the following additional types: {encountered_types}"
            )
        config_hash = custom_hash(self.get_params())
        if not all(custom_hash(v.get_params()) == config_hash for v in values):
            raise ValidationError(
                f"Based on the first value encountered for the scoring value {_key_name}, we expected all values to be "
                f"to be wrapped in an aggregator of type {type(self)} with the same configuration. "
                "However, we encountered at least one value with a different configuration. "
                f"Expected configuration should be {self.get_params()}"
            )

    def _get_emtpy_instance(self) -> Self:
        """Return an empty instance of the aggregator with the same config, but no value."""
        return self.clone()


class FloatAggregator(Aggregator[float]):
    def __init__(
        self, func: Callable[[Sequence[float]], Union[float, dict[str, float]]], *, return_raw_scores: bool = True
    ) -> None:
        self.func = func
        super().__init__(return_raw_scores=return_raw_scores)

    def aggregate(self, /, values: Sequence[float], datapoints: Sequence[Dataset]) -> Union[float, dict[str, float]]:  # noqa: ARG002
        """Aggregate a sequence of floats by taking the mean."""
        try:
            vals = self.func(values)
        except TypeError as e:
            raise ValidationError(
                f"Applying the float aggregation function {self.func} failed. "
                f"\n\n{values}"
            ) from e

        if isinstance(vals, dict):
            # We cast explicitly to float, to make sure that we get a float and not a float like
            return {k: float(v) for k, v in vals.items()}
        return float(vals)


mean_agg = FloatAggregator(np.mean)
mean_agg.__doc__ = """Aggregator that calculates the mean of the values."""


class _NoAgg(Aggregator[Any]):
    """Wrapper to wrap one or multiple output values of a scorer to prevent aggregation of these values.

    If one of the values in the return dictionary of a multi-value score function is wrapped with this class,
    the scorer will not aggregate the value.
    This allows to pass arbitary data from the score function through the scorer.
    As example, this could be some general metadata, some non-numeric scores, or an array of values (e.g. when the
    actual score is the mean of such values).

    Examples
    --------
    >>> def score_func(pipe, dataset):
    ...     ...
    ...     return {"score_val_1": score, "some_metadata": NoAgg(metadata)}
    >>> my_scorer = Scorer(score_func)

    """

    def aggregate(self, /, values: Sequence[Any], datapoints: Sequence[Dataset]) -> _Nothing:  # noqa: ARG002
        """Return nothing, indicating no aggregation."""
        return NOTHING


no_agg = _NoAgg()
no_agg.__doc__ = _NoAgg.__doc__


class Scorer(Generic[PipelineT, DatasetT, T], BaseTpcpObject):
    """A scorer to score multiple data points of a dataset and average the results.

    Parameters
    ----------
    score_func
        The callable that is used to score each data point
    single_score_callback
        Callback function that is called after each datapoint that is scored.
        It should have the following call signature:

        >>> def callback(
        ...     *, step: int, scores: Tuple[_SCORE_TYPE, ...], scorer: "Scorer", pipeline: Pipeline, dataset: Dataset, **_
        ... ) -> None:
        ...     ...

        All parameters will be passed as keyword arguments.
        This means, if your callback only needs a subset of the defined parameters, you can ignore them by using
        unused kwargs:

        >>> def callback(*, step: int, pipeline: Pipeline, **_):
        ...     ...

    n_jobs
        The number of parallel jobs to run.
        Each job will run on a single data point.
        Note, that the single_score_callback will still be called in the main thread, after a job is finished.
        However, it could be that multiple jobs are finished before the callback is called.
        The callback is still gurateed to be called in the order of the data points.
        If None, no parallelization is used.
    verbose
        Controls the verbosity of the parallelization.
        See :class:`joblib.Parallel` for more details.
    pre_dispatch
        Controls the number of jobs that get dispatched during parallelization.
        See :class:`joblib.Parallel` for more details.
    progress_bar
        True/False to enable/disable a `tqdm` progress bar.

    """

    score_func: ScoreFunc[PipelineT, DatasetT, T]
    default_aggregator: Aggregator
    single_score_callback: Optional[ScoreCallback[PipelineT, DatasetT, T]]
    n_jobs: Optional[int]
    verbose: int
    pre_dispatch: Union[str, int]
    progress_bar: bool

    def __init__(
        self,
        score_func: ScoreFunc[PipelineT, DatasetT, ScoreTypeT[T]],
        *,
        default_aggregator: Aggregator = cf(mean_agg),
        single_score_callback: Optional[ScoreCallback[PipelineT, DatasetT, T]] = None,
        # Multiprocess_kwargs
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        pre_dispatch: Union[str, int] = "2*n_jobs",
        progress_bar: bool = True,
    ) -> None:
        self.score_func = score_func
        self.default_aggregator = default_aggregator
        self.single_score_callback = single_score_callback
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.progress_bar = progress_bar

    # The typing for IndividualScoreType here is not perfect, but not sure how to fix.
    # For the aggregated scores, we can easily parameterize the value based on the generic, but not for the single
    # scores
    def __call__(
        self, pipeline: PipelineT, dataset: DatasetT
    ) -> tuple[Union[float, dict[str, float]], Union[Optional[list[T]], dict[str, list[T]]]]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all data-points
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, dataset=dataset)

    def _aggregate(
        self,
        scores: dict[str, tuple[Aggregator[T], list[T]]],
        datapoints: list[DatasetT],
    ) -> tuple[Union[float, dict[str, float]], Union[Optional[list[T]], dict[str, list[T]]]]:
        """Aggregate the scores."""
        is_single = len(scores) == 1 and "__single__" in scores
        if is_single and isinstance(scores["__single__"][0], _NoAgg):
            raise ValidationError(
                "Scorer returned a `no_agg` aggregator. "
                "This is not allowed when returning only a single score value. "
                "If you want to use a NoAgg scorer, return a dictionary of values, where one or "
                "multiple values are wrapped with NoAgg."
            )

        raw_scores: dict[str, list[T]] = {}
        agg_scores: dict[str, float] = {}
        for name, (aggregator, raw_score) in scores.items():
            if aggregator.return_raw_scores is True:
                raw_scores[name] = list(raw_score)
            try:
                agg_score = aggregator.aggregate(values=raw_score, datapoints=datapoints)
            except Exception as e:  # noqa: BLE001
                raise ValidationError(
                    f"Aggregator for score '{name}' raised an exception while aggregating scores. "
                    "Scroll up to see the original exception."
                ) from e
            if agg_score is NOTHING or isinstance(agg_score, _Nothing):
                # This is the case with the NoAgg Scorer
                continue
            if isinstance(agg_score, dict):
                # If the aggregator returned multiple values, we merge them prefixing the original name
                for key, value in agg_score.items():
                    agg_scores[key if is_single else f"{name}__{key}"] = value
            else:
                agg_scores[name] = agg_score
        # Finally we check that all aggregates values are floats
        if not all(isinstance(score, (int, float)) for score in agg_scores.values()):
            raise ValidationError(
                "Final aggregated scores are not all numbers. "
                "Double-check your (custom) aggregators."
                f"The current values are:\n{agg_scores}"
            )

        if is_single:
            # If we have only a single score, we return it directly
            return agg_scores.get("__single__", agg_scores), raw_scores.get("__single__", None)
        return agg_scores, raw_scores

    def _score(self, pipeline: PipelineT, dataset: DatasetT):
        def per_datapoint(i, d):
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                score = self.score_func(pipeline.clone(), d)
            except Exception as e:  # noqa: BLE001
                raise ScorerFailedError(
                    f"Scorer raised an exception while scoring data point {i} ({d.group_label}). "
                    "Tpcp does not support that (compared to sklearn) and you need to handle error cases yourself "
                    "within the scoring function."
                    "\n\n"
                    "The original exception was:\n\n"
                    f"{traceback.format_exc()}"
                ) from e
            return i, score

        scores: list[ScoreTypeT[T]] = []

        pbar = partial(tqdm, total=len(dataset), desc="Datapoints") if self.progress_bar else _passthrough
        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose, pre_dispatch=self.pre_dispatch, return_as="generator"
        )
        with parallel:
            for i, r in pbar(parallel(delayed(per_datapoint)(i, d) for i, d in enumerate(dataset))):
                scores.append(r)
                if self.single_score_callback:
                    self.single_score_callback(
                        step=i,
                        scores=tuple(scores),
                        scorer=self,
                        pipeline=pipeline,
                        dataset=dataset,
                    )

        return self._aggregate(_check_and_invert_score_dict(scores, self.default_aggregator), list(dataset))


ScorerTypes = Union[ScoreFunc[PipelineT, DatasetT, ScoreTypeT[T]], Scorer[PipelineT, DatasetT, ScoreTypeT[T]], None]


def _passthrough_scoring(pipeline: Pipeline[DatasetT], datapoint: DatasetT):
    """Call the score method of the pipeline to score the input."""
    return pipeline.score(datapoint)


def _validate_scorer(
    scoring: ScorerTypes[PipelineT, DatasetT, Any],
    pipeline: PipelineT,
    base_class: type[Scorer[Any, Any, Any]] = Scorer,
) -> Scorer[PipelineT, DatasetT, Any]:
    """Convert the provided scoring method into a valid scorer object."""
    if scoring is None:
        # If scoring is None, we will try to use the score method of the pipeline
        # However, we run score once with an empty dataset and check if it is actually implemented:
        try:
            pipeline.score(Dataset())  # type: ignore  # noqa: PGH003
        except NotImplementedError as e:
            raise e  # noqa: TRY201
        except Exception:  # noqa: BLE001
            pass
        scoring = _passthrough_scoring
    if isinstance(scoring, base_class):
        return scoring
    if callable(scoring):
        # We wrap the scorer, unless the user already supplied an instance of the Scorer class (or subclass)
        return base_class(scoring)
    raise ValueError("A valid scorer must either be a instance of `Scorer` (or subclass), None, or a callable.")


_non_homogeneous_scoring_error = ValidationError(
    "The returned score values are not homogeneous. "
    "For some datapoint a single values was returned, but for others a dictionary was returned. "
    "This is not allowed."
)


def _test_single_score_value(scores: list[any], default_agg: Aggregator):
    # We expect a single score value from each datapoint
    # We check that no other value is a dictionary.
    # Other than that we can check nothing else here.
    # What datatypes are really allowed is controlled by the aggregator and will be checked there.
    if any(isinstance(s, dict) for s in scores):
        raise _non_homogeneous_scoring_error
    if not isinstance(scores[0], Aggregator):
        # If the score is not wrapped in an aggregator, we wrap it in the default aggregator.
        scores = [default_agg.clone()(s) for s in scores]
    scores[0]._assert_is_all_valid(scores, "single score")
    return (
        scores[0].clone(),
        [s.get_value() for s in scores],
    )


def _invert_list_of_dicts(list_of_dicts: list[dict[str, Any]]) -> dict[str, list]:
    inverted = defaultdict(list)
    for d in list_of_dicts:
        if not isinstance(d, dict):
            raise _non_homogeneous_scoring_error
        for k, v in d.items():
            inverted[k].append(v)
    return dict(inverted)


def _check_and_invert_score_dict(
    #  I don't care that this is to complex, some things need to be complex
    scores: list[ScoreTypeT[T]],
    default_agg: Aggregator,
) -> dict[str, tuple[Aggregator[T], list[T]]]:
    """Invert the scores dictionary to a list of scores."""
    first_score = scores[0]
    if not isinstance(first_score, dict):
        return {"__single__": _test_single_score_value(scores, default_agg)}

    expected_length = len(scores)
    inverted_scoring_dict = _invert_list_of_dicts(scores)
    return_dict: dict[str, tuple[Aggregator[T], list[T]]] = {}
    for key, values in inverted_scoring_dict.items():
        if len(values) != expected_length:
            raise ValidationError(
                "The return values of the scoring function is expected to have the same keys for each datapoint. "
                f"However, for at least one datapoint the return values is missing the key '{key}'. "
                f"Expected keys are: {tuple(first_score.keys())}"
            )
        return_dict[key] = _test_single_score_value(values, default_agg)

    return return_dict
