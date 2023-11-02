import warnings
from dataclasses import fields, is_dataclass
from typing import Generic, Iterable, Iterator, List, Tuple, Type, TypeVar

from tpcp import Algorithm

DataclassT = TypeVar("DataclassT")
T = TypeVar("T")


class _NotSet:
    def __repr__(self):
        return "_NOT_SET"


_NOT_SET = _NotSet()


class TypedIterator(Algorithm, Generic[DataclassT, T]):
    _raw_results: List[DataclassT]
    done_: bool
    inputs_: List[T]

    def __init__(self, data_type: Type[DataclassT]):
        self.data_type = data_type

    def iterate(self, iterable: Iterable[T]) -> Iterator[Tuple[T, DataclassT]]:
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
        init_dict = {k.name: _NOT_SET for k in fields(self.data_type)}
        return self.data_type(**init_dict)

    @property
    def raw_results_(self) -> List[DataclassT]:
        if not self.done_:
            warnings.warn("The iterator is not done yet. The results might not be complete.", stacklevel=1)

        return self._raw_results

    def __getattr__(self, item):
        # We assume a correct result name ends with an underscore
        actual_item = item[:-1]

        if actual_item in self._raw_results[0].__dict__:
            return [getattr(r, actual_item) for r in self._raw_results]

        valid_result_fields = [k.name + "_" for k in fields(self.data_type)]

        raise AttributeError(
            f"Attribute {item} is not a valid attribute for {self.__class__.__name__} nor a dynamically generated "
            "result attribute of the result dataclass. "
            f"Valid result attributes are: {valid_result_fields}. "
            "Note the trailing underscore!"
        )
