import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import fields, is_dataclass
from typing import Any, Callable, Generic, TypeVar

from tpcp import Algorithm, cf

DataclassT = TypeVar("DataclassT")
T = TypeVar("T")


class _NotSet:
    def __repr__(self):
        return "_NOT_SET"


class TypedIterator(Algorithm, Generic[DataclassT]):
    """Helper to iterate over data and collect results.

    Parameters
    ----------
    data_type
        A dataclass that defines the result type you expect from each iteration.
    aggregations
        An optional list of aggregations to apply to the results.
        This has the form ``[(result_name, aggregation_function), ...]``.
        If a result-name is in the list, the aggregation will be applied to it, when accessing the respective result
        attribute (i.e. ``{result_name}_``).
        If no aggregation is defined for a result, a simple list of all results will be returned.
    NULL_VALUE
        (Class attribute) The value that is used to initialize the result dataclass and will remain in the results, if
        no result was for a specific attribute in one or more iterations.

    Attributes
    ----------
    inputs_
        List of all input elements that were iterated over.
    raw_results_
        List of all results as dataclass instances.
        The attribute of the dataclass instance will have the a value of ``_NOT_SET`` if no result was set.
        To check for this, you can use ``isinstance(val, TypedIterator.NULL_VALUE)``.
    {result_name}_
        The aggregated results for the respective result name.
    done_
        True, if the iterator is done.
        If the iterator is not done, but you try to access the results, a warning will be raised.

    """

    data_type: type[DataclassT]
    aggregations: Sequence[tuple[str, Callable[[list, list], Any]]]

    _raw_results: list[DataclassT]
    done_: bool
    inputs_: list

    NULL_VALUE = _NotSet()

    def __init__(
        self, data_type: type[DataclassT], aggregations: Sequence[tuple[str, Callable[[list, list], Any]]] = cf([])
    ):
        self.data_type = data_type
        self.aggregations = aggregations

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
        if not is_dataclass(self.data_type):
            raise TypeError(f"Expected a dataclass as data_type, got {self.data_type}")

        self._raw_results = []
        self.inputs_ = []
        self.done_ = False
        for d in iterable:
            result_object = self._get_new_empty_object()
            self._raw_results.append(result_object)
            self.inputs_.append(d)
            yield d, result_object
        self.done_ = True

    def _get_new_empty_object(self) -> DataclassT:
        init_dict = {k.name: self.NULL_VALUE for k in fields(self.data_type)}
        return self.data_type(**init_dict)

    @property
    def raw_results_(self) -> list[DataclassT]:
        if not self.done_:
            warnings.warn("The iterator is not done yet. The results might not be complete.", stacklevel=1)

        return self._raw_results

    def __getattr__(self, item):
        # We assume a correct result name ends with an underscore
        actual_item = item[:-1]

        if actual_item in self._raw_results[0].__dict__:
            values = [getattr(r, actual_item) for r in self.raw_results_]
            # if an aggregator is defined for the specific item, we apply it
            for name, aggregator in self.aggregations:
                if name == actual_item:
                    return aggregator(self.inputs_, values)
            return values

        valid_result_fields = [k.name + "_" for k in fields(self.data_type)]

        raise AttributeError(
            f"Attribute {item} is not a valid attribute for {self.__class__.__name__} nor a dynamically generated "
            "result attribute of the result dataclass. "
            f"Valid result attributes are: {valid_result_fields}. "
            "Note the trailing underscore!"
        )
