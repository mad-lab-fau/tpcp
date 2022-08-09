"""Helper to score pipelines."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast

import numpy as np
from typing_extensions import Protocol

from tpcp import NOTHING
from tpcp._base import _Nothing
from tpcp._dataset import Dataset, DatasetT
from tpcp._pipeline import Pipeline, PipelineT
from tpcp.exceptions import ScorerFailed, ValidationError

T = TypeVar("T")
AggReturnType = Union[float, Dict[str, float], _Nothing]

SingleScoreTypeT = Union[T, "Aggregator[Any]"]
MultiScoreTypeT = Dict[str, SingleScoreTypeT[T]]
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
        scores: Tuple[ScoreTypeT[T], ...],
        scorer: "Scorer[PipelineT, DatasetT, T]",
        pipeline: PipelineT,
        dataset: DatasetT,
    ) -> None:
        ...


class Aggregator(Generic[T]):
    """Base class for aggregators.

    You can subclass this class to create your own aggregators.
    The only thing you should change, is to overwrite the `aggregate` method.
    Everything else should not be modified.

    Custom aggregators can then be used to wrap return values of score functions or they can be passed as
    `default_aggregator` to the :class:`~tpcp.validate.Scorer` class.
    """

    _value: T

    def __init__(self, _value: T):
        self._value = _value

    def __repr__(self):
        """Show the representation of the object."""
        return f"{self.__class__.__name__}({repr(self._value)})"

    def get_value(self) -> T:
        """Return the value wrapped by aggregator."""
        return self._value

    @classmethod
    def aggregate(cls, /, values: Sequence[T], datapoints: Sequence[Dataset]) -> AggReturnType:
        """Aggregate the values."""
        raise NotImplementedError()


class MeanAggregator(Aggregator[float]):
    """Aggregator that calculates the mean of the values."""

    @classmethod
    def aggregate(cls, /, values: Sequence[float], datapoints: Sequence[Dataset]) -> float:
        """Aggregate a sequence of floats by taking the mean."""
        try:
            return float(np.mean(values))
        except TypeError as e:
            raise ValidationError(
                "MeanAggregator can only be used with float values. " f"Got the following values instead:\n\n{values}"
            ) from e


class NoAgg(Aggregator[Any]):
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

    @classmethod
    def aggregate(cls, /, values: Sequence[Any], datapoints: Sequence[Dataset]) -> _Nothing:  # noqa: unused-argument
        """Return nothing, indicating no aggregation."""
        return NOTHING


class Scorer(Generic[PipelineT, DatasetT, T]):
    """A scorer to score multiple data points of a dataset and average the results.

    Parameters
    ----------
    score_func
        The callable that is used to score each data point
    single_score_callback
        Callback function that is called after each datapoint that is scored.
        It should have the following call signature:

        >>> def callback(
        ...     *,
        ...     step: int,
        ...     scores: Tuple[_SCORE_TYPE, ...],
        ...     scorer: "Scorer",
        ...     pipeline: Pipeline,
        ...     dataset: Dataset,
        ...     **_
        ... ) -> None:
        ...     ...

        All parameters will be passed as keyword arguments.
        This means, if your callback only needs a subset of the defined parameters, you can ignore them by using
        unused kwargs:

        >>> def callback(*, step: int, pipeline: Pipeline, **_):
        ...     ...

    kwargs
        Additional arguments that might be used by the scorer.
        These are ignored for the base scorer.

    """

    kwargs: Dict[str, Any]
    _score_func: ScoreFunc[PipelineT, DatasetT, T]
    _single_score_func: Optional[ScoreCallback[PipelineT, DatasetT, T]]

    def __init__(
        self,
        score_func: ScoreFunc[PipelineT, DatasetT, ScoreTypeT[T]],
        *,
        default_aggregator: Type[Aggregator[T]] = MeanAggregator,
        single_score_callback: Optional[ScoreCallback[PipelineT, DatasetT, T]] = None,
        **kwargs: Any,
    ):
        self.kwargs = kwargs
        self._score_func = score_func
        self._default_aggregator = default_aggregator
        self._single_score_callback = single_score_callback

    # The typing for IndividualScoreType here is not perfect, but not sure how to fix.
    # For the aggregated scores, we can easily parameterize the value based on the generic, but not for the single
    # scores
    def __call__(
        self, pipeline: PipelineT, dataset: DatasetT
    ) -> Tuple[Union[float, Dict[str, float]], Union[List[Any], Dict[str, List[Any]]]]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all data-points
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, dataset=dataset)

    def _aggregate(  # noqa: no-self-use
        self,
        scores: Union[Tuple[Type[Aggregator[T]], List[T]], Dict[str, Tuple[Type[Aggregator[T]], List[T]]]],
        datapoints: List[DatasetT],
    ) -> Tuple[Union[float, Dict[str, float]], Union[List[T], Dict[str, List[T]]]]:
        if not isinstance(scores, dict):
            aggregator_single, raw_scores_single = scores
            if aggregator_single is NoAgg:
                raise ValidationError(
                    "Scorer returned a NoAgg object. "
                    "This is not allowed when returning only a single score value. "
                    "If you want to use a NoAgg scorer, return a dictionary of values, where one or "
                    "multiple values are wrapped with NoAgg."
                )
            # We create an instance of the aggregator here, even though we only need to call the class method.
            # This way, `aggregate` will work, even if people forgot to implement the aggregate method as class
            # method on their custom aggregator.
            agg_val = aggregator_single(None).aggregate(values=raw_scores_single, datapoints=datapoints)
            if isinstance(agg_val, dict):
                if not all(isinstance(score, float) for score in agg_val.values()):
                    raise ValidationError(
                        "Final aggregated scores are not all numbers. "
                        "Double-check your (custom) aggregators."
                        f"The current values are: \n{agg_val}"
                    )
            elif not isinstance(agg_val, (int, float)):
                raise ValidationError(f"The final aggregated score must be a numbers. Instead it is:\n{agg_val}")
            return agg_val, list(raw_scores_single)

        raw_scores: Dict[str, List[T]] = {}
        agg_scores: Dict[str, float] = {}
        for name, (aggregator, raw_score) in scores.items():
            raw_scores[name] = list(raw_score)
            agg_score = aggregator(None).aggregate(values=raw_score, datapoints=datapoints)
            if agg_score is NOTHING or isinstance(agg_score, _Nothing):
                # This is the case with the NoAgg Scorer
                continue
            if isinstance(agg_score, dict):
                # If the aggregator returned multiple values, we merge them prefixing the original name
                for key, value in agg_score.items():
                    agg_scores[f"{name}__{key}"] = value
            else:
                agg_scores[name] = agg_score
        # Finally we check that all aggregates values are floats
        if not all(isinstance(score, (int, float)) for score in agg_scores.values()):
            raise ValidationError(
                "Final aggregated scores are not all numbers. "
                "Double-check your (custom) aggregators."
                f"The current values are:\n{agg_scores}"
            )
        return agg_scores, raw_scores

    def _score(
        self, pipeline: PipelineT, dataset: DatasetT
    ) -> Tuple[Union[float, Dict[str, float]], Union[List[Any], Dict[str, List[Any]]]]:
        # `float` because the return value in case of an exception will always be float
        scores: List[ScoreTypeT[T]] = []
        datapoints: List[DatasetT] = []
        for i, d in enumerate(dataset):
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                score = self._score_func(pipeline.clone(), d)
            except Exception as e:  # noqa: broad-except
                raise ScorerFailed(
                    f"Scorer raised an exception while scoring data point {i}. "
                    "Tpcp does not support that (compared to sklearn) and you need to handle error cases yourself "
                    "within the scoring function."
                    "\n\n"
                    "The original exception was:\n\n"
                    f"{traceback.format_exc()}"
                ) from e

            scores.append(score)
            if self._single_score_callback:
                self._single_score_callback(
                    step=i,
                    scores=tuple(scores),
                    scorer=self,
                    pipeline=pipeline,
                    dataset=dataset,
                )
            datapoints.append(d)

        return self._aggregate(_check_and_invert_score_dict(scores, self._default_aggregator), datapoints)


ScorerTypes = Union[ScoreFunc[PipelineT, DatasetT, ScoreTypeT[T]], Scorer[PipelineT, DatasetT, ScoreTypeT[T]], None]


def _passthrough_scoring(pipeline: Pipeline[DatasetT], datapoint: DatasetT):
    """Call the score method of the pipeline to score the input."""
    return pipeline.score(datapoint)


def _validate_scorer(
    scoring: ScorerTypes[PipelineT, DatasetT, Any],
    pipeline: PipelineT,
    base_class: Type[Scorer[Any, Any, Any]] = Scorer,
) -> Scorer[PipelineT, DatasetT, Any]:
    """Convert the provided scoring method into a valid scorer object."""
    if scoring is None:
        # If scoring is None, we will try to use the score method of the pipeline
        # However, we run score once with an empty dataset and check if it is actually implemented:
        try:
            pipeline.score(Dataset())  # type: ignore
        except NotImplementedError as e:
            raise e
        except Exception:  # noqa: broad-except
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


def _check_and_invert_score_dict(  # noqa: MC0001  I don't care that this is to complex, some things need to be complex
    scores: List[ScoreTypeT[T]], default_agg: Type[Aggregator]
) -> Union[Tuple[Type[Aggregator[T]], List[T]], Dict[str, Tuple[Type[Aggregator[T]], List[T]]]]:
    """Invert the scores dictionary to a list of scores."""
    first_score = scores[0]
    if not isinstance(first_score, dict):
        # We expect a single score value from each datapoint
        # We check that no other value is a dictionary.
        # Other than that we can check nothing else here.
        # What datatypes are really allowed is controlled by the aggregator and will be checked there.
        if any(isinstance(s, dict) for s in scores):
            raise _non_homogeneous_scoring_error
        # We check that the score types are consistent for all datatypes.
        types = cast(
            Set[Type[Aggregator]], set((type(s) if isinstance(s, Aggregator) else default_agg) for s in scores)
        )
        if len(types) > 1:
            raise ValidationError(
                f"The score values are not consistent. The following aggregation types were found: {types}"
            )
        consistent_type = types.pop()
        return (
            consistent_type,
            [(s.get_value() if isinstance(s, consistent_type) else consistent_type(s).get_value()) for s in scores],
        )

    score_types = {k: (type(s) if isinstance(s, Aggregator) else default_agg) for k, s in first_score.items()}
    return_dict = {k: (v, []) for k, v in score_types.items()}
    for s in scores:
        if not isinstance(s, dict):
            raise _non_homogeneous_scoring_error
        # We check that the score types are consistent for all datatypes.
        for k, v in score_types.items():
            try:
                score = s[k]
            except KeyError as e:
                raise ValidationError(
                    "The return values of the scoring function is expected to have the same keys for each datapoint. "
                    f"However, for at least one datapoint the return values is missing the key '{k}'. "
                    f"Expected keys are: {tuple(score_types.keys())}"
                ) from e
            # If the score is not wrapped in an aggregator, we wrap it in the default aggregator.
            if not isinstance(score, Aggregator):
                # NOTE: Not really elegant to wrap the values and then get the value two lines below... But it works.
                score = default_agg(score)
            if not isinstance(score, v):
                raise ValidationError(
                    f"The score value for {k} is not consistently of type {v}."
                    f"For at least one datapoint, the score value was {s[k]}."
                )
            return_dict[k][1].append(score.get_value())

    return return_dict
