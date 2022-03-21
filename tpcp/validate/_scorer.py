"""Helper to score pipelines."""
from __future__ import annotations

import warnings
from traceback import format_exc
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NewType,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from typing_extensions import Protocol, TypeGuard

from tpcp._dataset import Dataset, Dataset_
from tpcp._pipeline import Pipeline, Pipeline_
from tpcp._utils._score import _ERROR_SCORE_TYPE, _SCORE_TYPE
from tpcp.exceptions import ScorerFailed

SingleScoreType = float
MultiScoreType = Dict[str, float]
ScoreType_ = TypeVar("ScoreType_", SingleScoreType, MultiScoreType)

IndividualScoreType = Union[Dict[str, List[float]], List[float]]

ScoreFuncSingle = Callable[[Pipeline_, Dataset_], SingleScoreType]
ScoreFuncMultiple = Callable[[Pipeline_, Dataset_], MultiScoreType]
ScoreFunc = Callable[[Pipeline_, Dataset_], ScoreType_]

ScoringError = NewType("ScoringError", float)


class ScoreCallback(Protocol[Pipeline_, Dataset_, ScoreType_]):
    """Callback signature for scorer callbacks."""

    def __call__(
        self,
        *,
        step: int,
        scores: Tuple[Union[ScoreType_, ScoringError], ...],
        scorer: "Scorer[Pipeline_, Dataset_, ScoreType_]",
        pipeline: Pipeline_,
        dataset: Dataset_,
        error_score: _ERROR_SCORE_TYPE,
    ) -> None:
        ...


class Scorer(Generic[Pipeline_, Dataset_, ScoreType_]):
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
        ...     error_score: _ERROR_SCORE_TYPE,
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
    _score_func: ScoreFunc[Pipeline_, Dataset_, ScoreType_]
    _single_score_func: Optional[ScoreCallback[Pipeline_, Dataset_, ScoreType_]]

    def __init__(
        self,
        score_func: ScoreFunc[Pipeline_, Dataset_, ScoreType_],
        *,
        single_score_callback: Optional[ScoreCallback[Pipeline_, Dataset_, ScoreType_]] = None,
        **kwargs: Any,
    ):
        self.kwargs = kwargs
        self._score_func = score_func
        self._single_score_callback = single_score_callback

    # The typing for IndividualScoreType here is not perfect, but not sure how to fix.
    # For the aggregated scores, we can easily parameterize the value based on the generic, but not for the single
    # scores
    def __call__(
        self, pipeline: Pipeline_, dataset: Dataset_, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[Union[ScoreType_, float], IndividualScoreType]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all data-points
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, dataset=dataset, error_score=error_score)

    def aggregate(self, scores: Sequence[float]) -> float:  # noqa: no-self-use
        """Aggregate the scores of each data point."""
        return float(np.mean(scores))

    def _score(
        self, pipeline: Pipeline_, dataset: Dataset_, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[Union[ScoreType_, float], IndividualScoreType]:
        scores: List[Union[ScoreType_, ScoringError]] = []
        for i, d in enumerate(dataset):
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                score = self._score_func(pipeline.clone(), d)
            except Exception:  # noqa: broad-except
                if error_score == "raise":
                    raise
                score = ScoringError(error_score)
                warnings.warn(
                    f"Scoring failed for data point: {d.groups}. "
                    f"The score of this data point will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    ScorerFailed,
                )
            # We check that the scorer returns only numeric values.
            _validate_score_return_val(score)
            scores.append(score)
            if self._single_score_callback:
                self._single_score_callback(
                    step=i,
                    scores=tuple(scores),
                    scorer=self,
                    pipeline=pipeline,
                    dataset=dataset,
                    error_score=error_score,
                )
        return aggregate_scores(scores, self.aggregate)  # type: ignore


ScorerTypes = Union[ScoreFunc[Pipeline_, Dataset_, ScoreType_], Scorer[Pipeline_, Dataset_, ScoreType_], None]


def _is_all_error(scores: List[Union[ScoreType_, ScoringError]], error_score: float) -> TypeGuard[List[float]]:
    return all([s == error_score for s in scores])


def _validate_score_return_val(value: Union[SingleScoreType, MultiScoreType, ScoringError]):
    """We expect a scorer to return either a numeric value or a dictionary of such values."""
    if isinstance(value, (int, float)):
        return
    if isinstance(value, dict):
        for v in value.values():
            if not isinstance(v, (int, float)):
                break
        else:
            return
    raise ValueError(
        "The scoring function must return either a dictionary of numeric values or a single numeric value."
    )


def _passthrough_scoring(pipeline: Pipeline[Dataset_], datapoint: Dataset_):
    """Call the score method of the pipeline to score the input."""
    return pipeline.score(datapoint)


def _validate_scorer(
    scoring: ScorerTypes[Pipeline_, Dataset_, Any],
    pipeline: Pipeline_,
    base_class: Type[Scorer[Any, Any, Any]] = Scorer,
) -> Scorer[Pipeline_, Dataset_, Any]:
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


@overload
def aggregate_scores(
    scores: List[Union[SingleScoreType, ScoringError]], agg_method: Callable[[Sequence[float]], float]
) -> Tuple[float, List[float]]:
    ...


@overload
def aggregate_scores(
    scores: List[Union[MultiScoreType, ScoringError]], agg_method: Callable[[Sequence[float]], float]
) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    ...


def aggregate_scores(
    scores: Union[List[Union[SingleScoreType, ScoringError]], List[Union[MultiScoreType, ScoringError]]],
    agg_method: Callable[[Sequence[float]], float],
) -> Union[Tuple[float, List[float]], Tuple[Dict[str, float], Dict[str, List[float]]]]:
    """Invert result dict of and apply aggregation method to each score output.

    Parameters
    ----------
    scores
        A list of either numeric values or dicts with numeric values. We expect all dicts to have the same structure
        in the latter case.
        If dicts and numeric values are mixed, we assume that the single numeric values should indicate a scoring
        error for this data point.
        In this case, the single value will be replaced by a dict having the same shape as all other dicts provided,
        but with all values being the value provided.
    agg_method
        A callable that can take a list of numeric values and returns a single value.
        It will be called on the list of scores provided.
        In case `scores` is a list of dicts, it will be called on the list of values

    Returns
    -------
    aggregated_scores
        If `scores` was a list of numeric values this will simply be the result of `agg_method(scores)`.
        If `scores` was a list of dicts, this will also be a dict, where `agg_method` was applied across all values
        with the respective dict key.
    single_scores
        If `scores` was a list of numeric values, this is simply `scores`.
        If `scores` was a list of dicts, this is the inverted dict (aka a dict of lists) with the original scores
        values.

    """
    # We need to go through all scores and check if one is a dictionary.
    # Otherwise, it might be possible that the values were caused by an error and hence did not return a dict as
    # expected.
    for s in scores:
        if isinstance(s, dict):
            score_names = s.keys()
            break
    else:
        scores = cast(List[float], scores)
        return agg_method(scores), scores
    inv_scores: Dict[str, List[float]] = {}
    agg_scores: Dict[str, float] = {}
    # Invert the dict and calculate the mean per score:
    for key in score_names:
        # If the scorer raised an error, there will only be a single value. This value will be used for all
        # scores then
        score_array = [score[key] if isinstance(score, dict) else score for score in scores]
        inv_scores[key] = score_array
        agg_scores[key] = agg_method(score_array)
    return agg_scores, inv_scores
