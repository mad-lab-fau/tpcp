"""Helper to score pipelines."""
from __future__ import annotations

import traceback
import warnings
from traceback import format_exc
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from tpcp._dataset import Dataset, DatasetT
from tpcp._pipeline import Pipeline, PipelineT
from tpcp.exceptions import ScorerFailed, ValidationError

T = TypeVar("T")

SingleScoreType = float
MultiScoreType = Dict[str, Union[float, "NoAgg"]]
ScoreType = Union[SingleScoreType, MultiScoreType]
ScoreTypeT = TypeVar("ScoreTypeT", SingleScoreType, MultiScoreType)

IndividualScoreType = Union[Dict[str, List[float]], List[float]]

ScoreFuncSingle = Callable[[PipelineT, DatasetT], SingleScoreType]
ScoreFuncMultiple = Callable[[PipelineT, DatasetT], MultiScoreType]
ScoreFunc = Callable[[PipelineT, DatasetT], ScoreTypeT]


class ScoreCallback(Protocol[PipelineT, DatasetT, ScoreTypeT]):
    """Callback signature for scorer callbacks."""

    def __call__(
        self,
        *,
        step: int,
        # The `float` in this call signature is there, because the
        # value will be float in case of a scoring error, independent of the remaining input types
        scores: Tuple[Union[ScoreTypeT, float], ...],
        scorer: "Scorer[PipelineT, DatasetT, ScoreTypeT]",
        pipeline: PipelineT,
        dataset: DatasetT,
    ) -> None:
        ...


class NoAgg(Generic[T]):
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

    _value: T

    def __init__(self, _value: T):
        self._value = _value

    def __repr__(self):
        """Show the represnentation of the object."""
        return f"{self.__class__.__name__}({repr(self._value)})"

    def get_value(self) -> T:
        """Return the value wrapped by NoAgg."""
        return self._value


class Scorer(Generic[PipelineT, DatasetT, ScoreTypeT]):
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
    _score_func: ScoreFunc[PipelineT, DatasetT, ScoreTypeT]
    _single_score_func: Optional[ScoreCallback[PipelineT, DatasetT, ScoreTypeT]]

    def __init__(
        self,
        score_func: ScoreFunc[PipelineT, DatasetT, ScoreTypeT],
        *,
        single_score_callback: Optional[ScoreCallback[PipelineT, DatasetT, ScoreTypeT]] = None,
        **kwargs: Any,
    ):
        self.kwargs = kwargs
        self._score_func = score_func
        self._single_score_callback = single_score_callback

    # The typing for IndividualScoreType here is not perfect, but not sure how to fix.
    # For the aggregated scores, we can easily parameterize the value based on the generic, but not for the single
    # scores
    def __call__(self, pipeline: PipelineT, dataset: DatasetT) -> Tuple[Union[ScoreTypeT, float], IndividualScoreType]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all data-points
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, dataset=dataset)

    def aggregate_single(  # noqa: no-self-use
        self, scores: Sequence[Union[float, NoAgg[T]]], name: Optional[str] = None
    ) -> Tuple[List[float, T], Optional[float]]:
        """Aggregate the scores of each data point.

        Parameters
        ----------
        scores
            The scores of each data point
        name
            The optional name of the score the values belong to.
            This will be None, if the scorer only handles a single value.

        """
        is_no_agg = False
        return_scores = []
        for score in scores:
            if isinstance(score, NoAgg):
                is_no_agg = True
                return_scores.append(score.get_value())
            else:
                return_scores.append(score)
        if is_no_agg:
            return return_scores, None
        return return_scores, float(np.mean(scores))

    def aggregate(
        self, scores: Union[Sequence[Union[float, NoAgg[T]]], Dict[str, Union[float, NoAgg[T]]]]
    ) -> Tuple[Union[ScoreTypeT, float], IndividualScoreType]:
        if not isinstance(scores, dict):
            raw_scores, agg_scores = self.aggregate_single(scores, None)
            if agg_scores is None:
                raise ValueError(
                    "Scorer returned a NoAgg object."
                    "This is not allowed when returning only a single score value."
                    "If you want to use a NoAgg scorer, return a dictionary of values, where one or "
                    "multiple values are wrapped with NoAgg."
                )
            return agg_scores, raw_scores

        raw_scores: Dict[str, List[float, T]] = {}
        agg_scores: Dict[str, float] = {}
        for name, score in scores.items():
            raw_score, agg_score = self.aggregate_single(score, name)
            raw_scores[name] = raw_score
            if agg_score is not None:
                agg_scores[name] = agg_score
        return agg_scores, raw_scores

    def _score(self, pipeline: PipelineT, dataset: DatasetT) -> Tuple[Union[ScoreTypeT, float], IndividualScoreType]:
        # `float` because the return value in case of an exception will always be float
        scores: List[Union[ScoreTypeT, float]] = []
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

            # We check that the scorer returns only numeric values.
            _validate_score_return_val(score)
            scores.append(score)
            if self._single_score_callback:
                self._single_score_callback(
                    step=i, scores=tuple(scores), scorer=self, pipeline=pipeline, dataset=dataset,
                )

        return self.aggregate(_invert_score_dict(scores))


ScorerTypes = Union[ScoreFunc[PipelineT, DatasetT, ScoreTypeT], Scorer[PipelineT, DatasetT, ScoreTypeT], None]


def _validate_score_return_val(value: ScoreType):
    """We expect a scorer to return either a numeric value or a dictionary of such values."""
    if isinstance(value, (int, float)):
        return
    if isinstance(value, dict):
        for v in value.values():
            if not isinstance(v, (int, float, NoAgg)):
                break
        else:
            return

    raise ValidationError(
        "The scoring function must have one of the following return types:\n"
        "1. dictionary of numeric values or values wrapped by `NoAgg`.\n"
        "2. single numeric value.\n\n"
        f"You return value was {value}"
    )


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


def _invert_score_dict(scores: List[ScoreType]) -> IndividualScoreType:
    """Invert the scores dictionary to a list of scores."""
    for s in scores:
        if isinstance(s, dict):
            score_names = s.keys()
            break
    else:
        return scores
    return_dict = {}
    for key in score_names:
        score_array = []
        for score in scores:
            if isinstance(score, dict):
                score_array.append(score[key])
            else:
                # If the scorer raised an error, there will only be a single value. This value will be used for all
                # scores then
                score_array.append(score)
        return_dict[key] = score_array
    return return_dict


# # Note, that there is an overlap between these call signatures which is a problem for typing.
# # In both cases it is valid that all values of `scores` are `float`.
# # This happens if an error is raised for all cases.
# # In this case it is not clear which call signature should be used (at least for the typechecker).
# # In reality this is equivalent to the "SingleScoreType" scenario.
# # Maybe there is a better way to type that in the future.
# @overload
# def aggregate_scores(
#     scores: List[Union[SingleScoreType, float]], agg_method: Callable[[Sequence[float]], float]
# ) -> Tuple[float, List[float]]:
#     ...
#
#
# @overload
# def aggregate_scores(
#     scores: List[Union[MultiScoreType, float]], agg_method: Callable[[Sequence[float]], float]
# ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
#     ...
#
#
# def aggregate_scores(
#     scores: Union[List[Union[SingleScoreType, float]], List[Union[MultiScoreType, float]]],
#     agg_method: Callable[[Sequence[float]], float],
# ) -> Union[Tuple[float, List[float]], Tuple[Dict[str, float], Dict[str, List[float]]]]:
#     """Invert result dict of and apply aggregation method to each score output.
#
#     Parameters
#     ----------
#     scores
#         A list of either numeric values or dicts with numeric values. We expect all dicts to have the same structure
#         in the latter case.
#         If dicts and numeric values are mixed, we assume that the single numeric values should indicate a scoring
#         error for this data point.
#         In this case, the single value will be replaced by a dict having the same shape as all other dicts provided,
#         but with all values being the value provided.
#     agg_method
#         A callable that can take a list of numeric values and returns a single value.
#         It will be called on the list of scores provided.
#         In case `scores` is a list of dicts, it will be called on the list of values
#
#     Returns
#     -------
#     aggregated_scores
#         If `scores` was a list of numeric values this will simply be the result of `agg_method(scores)`.
#         If `scores` was a list of dicts, this will also be a dict, where `agg_method` was applied across all values
#         with the respective dict key.
#     single_scores
#         If `scores` was a list of numeric values, this is simply `scores`.
#         If `scores` was a list of dicts, this is the inverted dict (aka a dict of lists) with the original scores
#         values.
#
#     """
#     # We need to go through all scores and check if one is a dictionary.
#     # Otherwise, it might be possible that the values were caused by an error and hence did not return a dict as
#     # expected.
#     for s in scores:
#         if isinstance(s, dict):
#             score_names = s.keys()
#             break
#     else:
#         scores = cast(List[float], scores)
#         return agg_method(scores), scores
#     inv_scores: Dict[str, List[float]] = {}
#     agg_scores: Dict[str, float] = {}
#     # Invert the dict and calculate the mean per score:
#     for key in score_names:
#         key_is_no_agg = False
#         score_array = []
#         for score in scores:
#             if isinstance(score, dict):
#                 score_val = score[key]
#                 if isinstance(score_val, NoAgg):
#                     # If one of the values are wrapped in NoAgg, we will not aggregate the values and only remove the
#                     # NoAgg warpper.
#                     key_is_no_agg = True
#                     score_array.append(score_val.get_value())
#                 else:
#                     score_array.append(score_val)
#             else:
#                 # If the scorer raised an error, there will only be a single value. This value will be used for all
#                 # scores then
#                 score_array.append(score)
#         inv_scores[key] = score_array
#         if not key_is_no_agg:
#             agg_scores[key] = agg_method(score_array)
#     return agg_scores, inv_scores
