import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

from typing_extensions import NamedTuple, TypeAlias

from tpcp import Algorithm, cf

DataclassT = TypeVar("DataclassT")
InputTypeT = TypeVar("InputTypeT")
ResultT = TypeVar("ResultT")
T = TypeVar("T")


class TypedIteratorResultTuple(NamedTuple, Generic[InputTypeT, ResultT]):
    iteration_name: str
    input: InputTypeT
    result: ResultT
    iteration_context: dict[str, Any]


class _NotSet:
    def __repr__(self) -> str:
        return "NOT_SET"


_NULL_VALUE = _NotSet()


class BaseTypedIterator(Algorithm, Generic[InputTypeT, DataclassT]):
    """A Base class to implement custom typed iterators.

    This class is missing the ``iterate`` method, which needs to be implemented by the child class.
    It has a `_iterate` method though that does most of the heavy lifting.
    The actual iterate method should handle turning the inputs into an iterable and then call `_iterate`.

    :class:`TypedIterator` provides a "dummy" implementation that expects any type of iterator for the iterate method.
    Custom base classes could provide more elaborate preprocessing of the inputs before iteration.
    For example, cutting sections out of a dataframe based on a list of start and end indices, and then iterating over
    the cut sections.

    Parameters
    ----------
    data_type
        A dataclass that defines the result type you expect from each iteration.
    aggregations
        An optional list of aggregations to apply to the results.
        This has the form ``[(result_name, aggregation_function), ...]``.
        Each aggregation function gets ``raw_results_`` provided as input and can return an arbitrary object.
        Note, that the aggregation function needs to handle the case where no result was set for a specific attribute.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the ``results_``
        (i.e. ``results_.{result_name}``).
        If no aggregation is defined for a result, a simple list of all results will be returned.
        .. note:: It is possible to define aggregations with names not present in the result dataclass.
                  This allows to provide multiple aggregations from the same data.
                  The results can still be accessed via the ``additional_results_`` attribute.
    NULL_VALUE
        (Class attribute) The value that is used to initialize the result dataclass and will remain in the results, if
        no result was for a specific attribute in one or more iterations.
    IteratorResult
        (Class attribute) Type alias for the result-type of the iterator. ``raw_results_`` will be a list of these.
        Note, that when using this outside of the class, this type will be a generic without a type for the ``input``
        and ``result`` field.

    Attributes
    ----------
    results_
        The actual aggregated results.
        This is an instance of the provided custom result type (``data_type``).
        This makes it easy to access the individual result attributes.
        Each value will be the result of the aggregation function registered for the specific field.
        If no aggregation functions exist, simply a list of the result values for this field is provided.
        The results can only be accessed once the iteration is done.
    additional_results_
        A dictionary with additional results that were created by aggregators with names not present in the result
        dataclass.
    raw_results_
        List of all results as ``TypedIteratorResultTuple`` instances.
        This is the input to the aggregation functions.
        The attribute of the ``result`` dataclass instance will have the value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, BaseTypedIterator.NULL_VALUE)`` or the
        ``BaseTypedIterator.filter_iterator_results`` method to remove all results with a ``NULL_VALUE``.
    done_
        A dictionary indicating of a specific iterator is done.
        This usually only has the key ``__main__`` for the main iteration triggered by ``iterate``.
        However, subclasses can define nested iterations with more complex logic.
        The value will be ``True`` if the respective iteration is done, ``False`` if it is currently running and
        missing if it was never started.
        If the main iterator is not done, but you try to access the results, an error will be raised.

    Notes
    -----
    Under the hood, the iterator supports having multiple iterations at the same time by providing an ``iteration_name``
    to the ``_iterate`` method.
    This information is stored in the results and can be used to separate the results of different iterations.
    Together with the ``iteration_context`` parameter, this allows for more complex iteration structures.
    One example, would be the use of nested iterations, that are aware of the parent iteration.

    In the ``mobilise-d/mobgap`` library this is used to support the iteration of multiple levels of interests within
    data.
    For example, the outer level iterates Gait-Tests, the inner level iterates gait sequences within each gait test
    that are dynamically detected within the outer iteration.
    The iterator can then still patch all results from the inner iteration together to provide a single result object
    with times that are relative to the start of the entire recording.

    """

    IteratorResult: TypeAlias = TypedIteratorResultTuple[InputTypeT, DataclassT]

    _raw_results: list[IteratorResult]
    data_type: type[DataclassT]
    aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]]

    _result_fields: set[str]
    # We use this as cache
    _results: DataclassT
    _additional_results: dict[str, Any]

    done_: dict[str, bool]

    NULL_VALUE = _NULL_VALUE

    def __init__(
        self,
        data_type: type[DataclassT],
        aggregations: Sequence[tuple[str, Callable[[list[IteratorResult]], Any]]] = cf([]),
    ) -> None:
        self.data_type = data_type
        self.aggregations = aggregations

    def _iterate(
        self,
        iterable: Iterable[T],
        *,
        iteration_name: str = "__main__",
        iteration_context: Optional[dict[str, Any]] = None,
    ) -> Iterator[tuple[T, DataclassT]]:
        """Iterate over the given iterable and yield the input and a new empty result object for each iteration.

        Parameters
        ----------
        iterable
            The iterable to iterate over.
            Note, that the iterable is expected to yield a tuple of the form `(unique_iteration_id, value)` for each
            iteration.
            The iteration id is used internally to keep track of what results belong to which iteration.
        iteration_name
            The name of the iteration.
            This is an advanced feature and should only be used if you want to have multiple nested iterations.
            See the iterator example for potential usecases.
        iteration_context
            This is any piece of information that you want to pass to all aggregation functions via the result object.

        Yields
        ------
        input, result_object
            The input and a new empty result object.
            The result object is a dataclass instance of the type defined in ``self.data_type``.
            All values of the result object are set to ``TypedIterator.NULL_VALUE`` by default.

        """
        if not is_dataclass(self.data_type):
            raise TypeError(f"Expected a dataclass as data_type, got {self.data_type}")

        if iteration_name == "__main__":
            # Reset all caches
            if hasattr(self, "_results"):
                del self._results
            if hasattr(self, "_add"):
                del self._additional_results
            self.done_ = {}
            self._raw_results = []

        self.done_[iteration_name] = False
        for d in iterable:
            result_object = self._get_new_empty_object()
            result_tuple = TypedIteratorResultTuple(iteration_name, d, result_object, iteration_context or {})
            self._report_new_result(result_tuple)
            yield d, result_object
        self.done_[iteration_name] = True

    def _get_new_empty_object(self) -> DataclassT:
        init_dict = {k.name: self.NULL_VALUE for k in fields(self.data_type)}
        return self.data_type(**init_dict)

    def _report_new_result(self, r: TypedIteratorResultTuple[InputTypeT, DataclassT]) -> None:
        self._raw_results.append(r)

    @property
    def raw_results_(self) -> list[IteratorResult]:
        if "__main__" not in self.done_:
            raise ValueError(
                "The iterator has not been started yet. No results are available. Call the iterate method first."
            )
        if not self.done_["__main__"]:
            warnings.warn("The iterator is not done yet. The results might not be complete.", stacklevel=1)

        return self._raw_results

    def _get_default_agg(self, field_name: str) -> Callable[[list[IteratorResult]], Any]:
        def default_agg(values: list[BaseTypedIterator.IteratorResult]) -> list[Any]:
            return [getattr(v.result, field_name) for v in values]

        return default_agg

    def _agg_result(self, raw_results: list[IteratorResult]) -> tuple[dict[str, Any], dict[str, Any]]:
        aggregations = dict(self.aggregations)
        agg_results = {}
        for a in fields(self.data_type):
            # if an aggregator is defined for the specific item, we apply it
            name = a.name
            if name in aggregations:
                values = aggregations.pop(name)(raw_results)
            else:
                values = self._get_default_agg(name)(raw_results)
            agg_results[name] = values
        # If there are further aggregations, we apply them as well
        additional_aggregations = {
            name: agg(raw_results) for name, agg in aggregations.items() if name not in agg_results
        }
        return agg_results, additional_aggregations

    def _cache_agg(self) -> None:
        agg_results, additional_aggregations = self._agg_result(self.raw_results_)
        self._results = self.data_type(**agg_results)
        self._additional_results = additional_aggregations

    @property
    def results_(self) -> DataclassT:
        """The aggregated results.

        Note, that this returns an instance of the result object, even-though the datatypes of the attributes might be
        different depending on the aggregation function.
        We still decided it makes sense to return an instance of the result object, as it will allow to autocomplete
        the attributes, even-though the associated times might not be correct.
        """
        if not hasattr(self, "_results"):
            self._cache_agg()
        return self._results

    @property
    def additional_results_(self) -> dict[str, Any]:
        if not hasattr(self, "_additional_results"):
            self._cache_agg()
        return self._additional_results

    @classmethod
    def filter_iterator_results(
        cls,
        values: list[IteratorResult[Any, Any]],
        result_name: str,
        _null_value: _NotSet = _NULL_VALUE,
    ) -> list[IteratorResult[Any, Any]]:
        return [v._replace(result=r) for v in values if (r := getattr(v.result, result_name)) is not _null_value]


class TypedIterator(BaseTypedIterator[InputTypeT, DataclassT], Generic[InputTypeT, DataclassT]):
    """Helper to iterate over data and collect results.

    Parameters
    ----------
    data_type
        A dataclass that defines the result type you expect from each iteration.
    aggregations
        An optional list of aggregations to apply to the results.
        This has the form ``[(result_name, aggregation_function), ...]``.
        Each aggregation function gets ``raw_results_`` provided as input and can return an arbitrary object.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the ``results_``
        (i.e. ``results_.{result_name}``).
        If no aggregation is defined for a result, a simple list of all results will be returned.
        .. note:: It is possible to define aggregations with names not present in the result dataclass.
                  This allows to provide multiple aggregations from the same data.
                  The results can still be accessed via the ``additional_results_`` attribute.
    NULL_VALUE
        (Class attribute) The value that is used to initialize the result dataclass and will remain in the results, if
        no result was for a specific attribute in one or more iterations.
    IteratorResult
        (Class attribute) Type alias for the result-type of the iterator. ``raw_results_`` will be a list of these.
        Note, that when using this outside of the class, this type will be a generic without a type for the ``input``
        and ``result`` field.

    Attributes
    ----------
    results_
        The actual aggregated results.
        This is an instance of the provided custom result type (``data_type``).
        This makes it easy to access the individual result attributes.
        Each value will be the result of the aggregation function registered for the specific field.
        If no aggregation functions exist, simply a list of the result values for this field is provided.
        The results can only be accessed once the iteration is done.
    additional_results_
        A dictionary with additional results that were created by aggregators with names not present in the result
        dataclass.
    raw_results_
        List of all results as ``TypedIteratorResultTuple`` instances.
        This is the input to the aggregation functions.
        The attribute of the ``result`` dataclass instance will have the value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, TypedIterator.NULL_VALUE)`` or the
        ``TypedIterator.filter_iterator_results`` method to remove all results with a ``NULL_VALUE``.
    done_
        A dictionary indicating of a specific iterator is done.
        This usually only has the key ``__main__`` for the main iteration triggered by ``iterate``.
        However, subclasses can define nested iterations with more complex logic.
        The value will be ``True`` if the respective iteration is done, ``False`` if it is currently running and
        missing if it was never started.
        If the main iterator is not done, but you try to access the results, an error will be raised.

    """

    def iterate(self, iterable: Iterable[T]) -> Iterator[tuple[T, DataclassT]]:
        """Iterate over the given iterable and yield the input and a new empty result object for each iteration.

        Parameters
        ----------
        iterable
            The iterable to iterate over.

        Yields
        ------
        input, result_object
            The input and a new empty result object.
            The result object is a dataclass instance of the type defined in ``self.data_type``.
            All values of the result object are set to ``TypedIterator.NULL_VALUE`` by default.

        """
        yield from self._iterate(iterable)
