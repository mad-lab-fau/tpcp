"""Helper to score pipelines."""
from __future__ import annotations

import numbers
import warnings
from traceback import format_exc
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type, Union

import numpy as np

from tpcp.dataset import Dataset
from tpcp._utils._score import _ERROR_SCORE_TYPE, _AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE

if TYPE_CHECKING:
    from tpcp.pipelines import SimplePipeline


class Scorer:
    """A scorer to score multiple data points of a dataset and average the results.

    Parameters
    ----------
    score_func
        The callable that is used to score each data point
    kwargs
        Additional arguments that might be used by the scorer.
        These are ignored for the base scorer.

    """

    def __init__(self, score_func: Callable, **kwargs):
        self._kwargs = kwargs
        self._score_func = score_func

    def __call__(
        self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
        """Score the pipeline with the provided data.

        Returns
        -------
        agg_scores
            The average scores over all data-points
        single_scores
            The scores for each individual data-point

        """
        return self._score(pipeline=pipeline, data=data, error_score=error_score)

    def aggregate(self, scores: np.ndarray) -> float:  # noqa: no-self-use
        """Aggregate the scores of each data point."""
        return float(np.mean(scores))

    def _score(
        self, pipeline: SimplePipeline, data: Dataset, error_score: _ERROR_SCORE_TYPE
    ) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
        scores = []
        for d in data:
            try:
                # We need to clone here again, to make sure that the run for each data point is truly independent.
                score = self._score_func(pipeline.clone(), d)
            except Exception:  # noqa: broad-except
                if error_score == "raise":
                    raise
                score = error_score
                warnings.warn(
                    f"Scoring failed for data point: {d.groups}. "
                    f"The score of this data point will be set to {error_score}. Details: \n"
                    f"{format_exc()}",
                    UserWarning,
                )
            # We check that the scorer returns only numeric values.
            _validate_score_return_val(score)
            scores.append(score)
        return _aggregate_scores(scores, self.aggregate)


def _validate_score_return_val(value: _SINGLE_SCORE_TYPE):
    """We expect a scorer to return either a numeric value or a dictionary of such values."""
    if isinstance(value, numbers.Number):
        return
    if isinstance(value, dict):
        for v in value.values():
            if not isinstance(v, numbers.Number):
                break
        else:
            return
    raise ValueError(
        "The scoring function must return either a dictionary of numeric values or a single numeric value."
    )


def _passthrough_scoring(pipeline: SimplePipeline, datapoint: Dataset):
    """Call the score method of the pipeline to score the input."""
    return pipeline.score(datapoint)


def _validate_scorer(
    scoring: Optional[Union[Callable, Scorer]],
    pipeline: SimplePipeline,
    base_class: Type[Scorer] = Scorer,
) -> Scorer:
    """Convert the provided scoring method into a valid scorer object."""
    if scoring is None:
        # If scoring is None, we will try to use the score method of the pipeline
        # However, we run score once with an empty dataset and check if it is actually implemented:
        try:
            pipeline.score(Dataset())
        except NotImplementedError as e:
            raise e
        except Exception:  # noqa: broad-except
            pass
        scoring = _passthrough_scoring
    if isinstance(scoring, base_class):
        return scoring
    if callable(scoring):
        # We wrap the scorer, unless the user already supplied a instance of the Scorer class (or subclass)
        return base_class(scoring)
    raise ValueError("A valid scorer must either be a instance of `Scorer` (or subclass), None, or a callable.")


def _aggregate_scores(scores: _SCORE_TYPE, agg_method: Callable) -> Tuple[_AGG_SCORE_TYPE, _SINGLE_SCORE_TYPE]:
    """Invert result dict of and apply aggregation method to each score output."""
    # We need to go through all scores and check if one is a dictionary.
    # Otherwise it might be possible that the values were caused by an error and hence did not return a dict as
    # expected.
    for s in scores:
        if isinstance(s, dict):
            score_names = s.keys()
            break
    else:
        return agg_method(scores), np.asarray(scores)
    inv_scores = {}
    agg_scores = {}
    # Invert the dict and calculate the mean per score:
    for key in score_names:
        # If the the scorer raised an error, there will only be a single value. This value will be used for all
        # scores then
        score_array = np.asarray([score[key] if isinstance(score, dict) else score for score in scores])
        inv_scores[key] = score_array
        agg_scores[key] = agg_method(score_array)
    return agg_scores, inv_scores
