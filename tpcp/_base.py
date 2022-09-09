"""Private base classes for tpcp.

These classes are in a separate module to avoid circular imports.
In basically all cases, you do not need them.
"""
from __future__ import annotations

import copy
import dataclasses
import inspect
import sys
import warnings
from collections import defaultdict
from functools import wraps
from types import MethodWrapperType
from typing import (
    Any,
    Callable,
    ClassVar,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing_extensions import Annotated, Literal, Self, get_args, get_origin

from tpcp._parameters import _ParaTypes
from tpcp.exceptions import MutableDefaultsError, PotentialUserErrorWarning, ValidationError

T = TypeVar("T")
BaseTpcpObjectObjT = TypeVar("BaseTpcpObjectObjT", bound="BaseTpcpObject")


class _Nothing:
    """Sentinel class to indicate the lack of a value when ``None`` is ambiguous.

    ``_Nothing`` is a singleton. There is only ever one of it.

    This implementation is taken from the attrs package.
    """

    _singleton: Optional[_Nothing] = None

    def __new__(cls) -> _Nothing:
        if _Nothing._singleton is None:
            _Nothing._singleton = super(_Nothing, cls).__new__(cls)
        return _Nothing._singleton

    def __repr__(self) -> str:
        return "NOTHING"

    def __bool__(self) -> Literal[False]:
        return False

    def __len__(self) -> Literal[0]:
        return 0  # __bool__ for Python 2


NOTHING = _Nothing()
"""
Sentinel to indicate the lack of a value when ``None`` is ambiguous.
"""


def _get_init_defaults(cls: Type[_BaseTpcpObject]) -> Dict[str, inspect.Parameter]:
    # fetch the constructor or the original constructor before deprecation wrapping if any
    init = cls.__init__
    if init is object.__init__ or isinstance(init, MethodWrapperType):
        # No explicit constructor to introspect
        return {}

    # introspect the constructor arguments to find the model parameters to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    defaults = {k: p for k, p in init_signature.parameters.items() if p.name != "self" and p.kind != p.VAR_KEYWORD}
    return defaults


def _replace_defaults_wrapper(old_init: Callable) -> Callable:
    """Decorate an init to create new instances of mutable defaults.

    This should only be used in combination with `default` and will be applied as part of `__init_subclass`.
    Direct usage of this decorator should not be required.
    """

    @wraps(old_init)
    def new_init(self: BaseTpcpObject, *args: Any, **kwargs: Any) -> None:
        # call the old init.
        old_init(self, *args, **kwargs)
        self.__post_init__()

    return new_init


def _retry_eval_with_missing_locals(
    expression: str, globalns: Optional[Dict[str, Any]] = None, localns: Optional[Dict[str, Any]] = None
) -> Any:
    globalns = globalns or {}
    localns = localns or {}
    # We make a copy to not overwrite the input dict
    localns = {**localns}

    # That seems scary :D Let's see if this causes any issues
    # We use a value here instead of a "while True" to not get the program stuck in an endless loop.
    for _ in range(100):
        try:
            val = eval(expression, globalns, localns)  # noqa: eval-used
            break
        except NameError as e:
            missing = str(e).split("'")[1]
            localns[missing] = None
        except AttributeError as e:
            raise RuntimeError(
                "You ran into an edge case of the builtin type resolver. "
                "This happens if you use a nested type annotation that is only valid during runtime. "
                "This usually happens if you are using a `if TYPE_CHECKING:` guard for some of your imports to avoid "
                "circular dependencies.\n"
                "For most of these cases we have a built in workaround, but only if you provide the type directly and "
                "not in an attribute notation:\n"
                "\n"
                "This is not allowed:"
                "\n"
                ">>> if TYPE_CHECKING:\n"
                "...     import my\n"
                ">>> MyClass:\n"
                "...     para: my.custom_type\n"
                "\n\n"
                "This should work:\n"
                ">>> if TYPE_CHECKING:\n"
                "...     from my import custom_type\n"
                ">>> MyClass:\n"
                "...     para: custom_type\n"
                "\n\n"
                "And this as well:\n"
                ">>> if TYPE_CHECKING:\n"
                "...     import my\n"
                "...     custom_type_var = my.custom_type\n"
                ">>> MyClass:\n"
                "...     para: custom_type_var\n"
            ) from e
    else:
        raise RuntimeError(
            "Trying to resolve Parameter Type hints has resulted in an unexpected issue. "
            f"We were trying to evaluate the expression: {expression} . "
            "It seems like this expression contained forward references that could not be resolved. "
            "This should not happen!"
            "Please open an issue on GitHub with an code example that results in this issue. "
            "Sry for the inconvenience :)"
        )
    return val


def _custom_get_type_hints(cls: Type[_BaseTpcpObject]) -> Dict[str, Any]:
    """Extract type hints while avoiding issues with forward references."""
    hints = {}
    for base in reversed(cls.__mro__):
        base_globals = sys.modules[base.__module__].__dict__
        ann = base.__dict__.get("__annotations__", {})
        for name, value in ann.items():
            if value is None:
                value = type(None)
            elif isinstance(value, str):
                # TODO: This does not check if the str is a valid expression.
                #   This might not be an issue, but could lead to obscure error messages.
                value = _retry_eval_with_missing_locals(value, base_globals)
            hints[name] = value
    return hints


def _extract_partial_annotations(cls: Type[_BaseTpcpObject]) -> Dict[str, _ParaTypes]:
    cls_annotations = _custom_get_type_hints(cls)
    para_annotations = {}
    for k, v in cls_annotations.items():
        origin = get_origin(v)
        if origin is ClassVar:
            # If the parameter is a ClassVar, we go one level deeper and check, if its argument was annotated.
            v = get_args(v)[0]
            origin = get_origin(v)
        if origin is Annotated:
            for annot in get_args(v)[1:]:
                if isinstance(annot, _ParaTypes):
                    para_annotations[k] = annot
                    break

    return para_annotations


def _add_missing_init_fields_to_annotations(
    para_annotations: Dict[str, _ParaTypes], init_fields: Dict[str, inspect.Parameter]
):
    for k in init_fields:
        if k not in para_annotations:
            para_annotations[k] = _ParaTypes.SIMPLE


def _run_field_checks(cls, fields: Dict[str, Any]) -> None:
    _has_dangerous_mutable_default(fields, cls)
    _has_invalid_name(fields, cls)
    _annotations_are_valid(fields, cls)


def _process_tpcp_class(cls: Type[_BaseTpcpObject]):
    # We extract the fields of the init
    fields = _get_init_defaults(cls)

    # Now that we know the init fields for sure, we can complete our parameter extraction
    _add_missing_init_fields_to_annotations(cls.__field_annotations__, fields)

    # Validation
    if cls.__skip_validation__ is not True:
        _run_field_checks(cls, fields)

    if dataclasses.is_dataclass(cls) and any(isinstance(field.default, BaseFactory) for field in fields.values()):
        # If the class is already a data class when this check runs (aka when it is triggered by the custom
        # decorator), then we don't need to do anything.
        # The BaseTpcpObject already implements the post_init method to handly all factory stuff.
        # However, when you are already using dataclasses, you should probably use their factory methods.
        warnings.warn(
            "You are using the tpcp default factory (`cf`/`CloneFactory`) in combination with "
            "`dataclasses`. "
            "Consider using the `factory` option of `dataclasses.field` instead."
        )

    cls.__tpcp_cls_processed__ = True


def _requires_processed_class(instance: _BaseTpcpObject, method_name: str) -> None:
    if not instance.__tpcp_cls_processed__:
        if not instance.__field_annotations__ and type(instance).__init__ == object.__init__:
            # If there are no annotations and no init, it was just a simple class without parameters or init.
            # This is fine and we mark it as processed to speed up future checks.
            type(instance).__tpcp_cls_processed__ = True
            return
        raise RuntimeError(
            f"You are calling {method_name} on a class that has not been processed by tpcp. "
            "This should not happen!\n\n"
            "The only case we know of, is when you defined a class with class annotations, but without an init or a"
            "dataclass decorator. "
            "This is likely a user error.\n\n"
            "If this is not what happened, please open an issue on GitHub with an example explaining how you "
            "reached this error."
        )


class _BaseTpcpObject:
    __field_annotations__: Dict[str, _ParaTypes]
    __skip_validation__: bool
    __tpcp_cls_processed__: bool

    def __init_subclass__(cls, *, _skip_validation: bool = False, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls.__skip_validation__ = _skip_validation
        # We set that here to make sure the value is not inherited from the parent class.
        cls.__tpcp_cls_processed__ = False
        cls.__tpcp_might_have_empty_init__ = False

        # Then we extract all custom annotations that have been made.
        # We can do that safely, even before a dataclass decorator might process the class
        cls.__field_annotations__ = _extract_partial_annotations(cls)

        # If the class has no init or is a dataclass (aka a subclass of a dataclass), we don't need to do anything.
        # It could be that it is a dataclass, but then the dataclass wrapper will already call the post_init
        # correctly.
        #
        # If our class simply does not have an __init__ and will not get one via the dataclass wrapper, there is
        # nothing to do anyway.
        if cls.__init__ is object.__init__ or dataclasses.is_dataclass(cls):
            return

        # If we have a custom init, we need to wrap it to call the post_init method.
        setattr(cls, "__init__", _replace_defaults_wrapper(cls.__init__))


class BaseTpcpObject(_BaseTpcpObject, _skip_validation=True):
    """Baseclass for all tpcp objects."""

    def get_params(self: Self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this algorithm.

        Parameters
        ----------
        deep
            Only relevant if object contains nested algorithm objects.
            If this is the case and deep is True, the params of these nested objects are included in the output using a
            prefix like `nested_object_name__` (Note the two "_" at the end)

        Returns
        -------
        params
            Parameter names mapped to their values.

        """
        return _get_params(self, deep)

    def set_params(self: Self, **params: Any) -> Self:
        """Set the parameters of this Algorithm.

        To set parameters of nested objects use `nested_object_name__para_name=`.
        """
        return _set_params(self, **params)

    def clone(self) -> Self:
        """Create a new instance of the class with all parameters copied over.

        This will create a new instance of the class itself and all nested objects
        """
        return clone(self, safe=True)

    def __repr__(self):
        """Provide generic representation for the object based on all parameters."""
        class_name = type(self).__name__
        paras = self.get_params(deep=False)
        result = [class_name, "("]
        first = True
        for name, para in paras.items():
            if first:
                first = False
            else:
                result.append(", ")
            result.extend((name, "=", repr(para)))
        return "".join(result) + ")"

    def __post_init__(self):
        """Replace all parameters that use the `CloneFactory` with the actual values.

        This will either be called by `dataclasses.dataclass` or by in the modified `__init__` tpcp creates for
        normal classes.
        """
        # When using dataclasses, this method will be called, even if our custom processing did not run.
        # However, this is an issue, and we need to force users to apply the custom decorator.
        if not self.__tpcp_cls_processed__:
            # The first time we run an init of this class we run our processing.
            # Running that in the init for the first time, makes sure that it runs after any processing run on the class
            # itself (i.e. when a dataclass decorator is used).
            _process_tpcp_class(type(self))

        # Check if any of the initial values has a "default parameter flag".
        # If yes we replace it with a clone (in case of a tpcp object) or a deepcopy in case of other objects.
        # This is handled by the factory `get_value` method.
        for k, v in self.get_params(deep=False).items():
            if isinstance(v, BaseFactory):
                setattr(self, k, v.get_value())


# TODO: Make public api
def _get_params(instance: _BaseTpcpObject, deep: bool = True) -> Dict[str, Any]:
    _requires_processed_class(instance, "get_params")
    valid_fields = get_param_names(type(instance))

    out: Dict[str, Any] = {}
    for key in valid_fields:
        value = getattr(instance, key)
        # This is a little magic, that also gets the parameters of nested sklearn classifiers.
        if deep and hasattr(value, "get_params"):
            deep_items = value.get_params(deep=True).items()
            out.update((key + "__" + k, val) for k, val in deep_items)
        out[key] = value
    return out


def _set_params(instance: BaseTpcpObjectObjT, **params: Any) -> BaseTpcpObjectObjT:
    """Set the parameters of an instance.

    To set parameters of nested objects use `nested_object_name__para_name=`.
    """
    # Basically copied from sklearn
    if not params:
        return instance
    valid_params = instance.get_params(deep=True)

    nested_params: DefaultDict[str, Any] = defaultdict(dict)  # grouped by prefix
    for key, value in params.items():
        key, delim, sub_key = key.partition("__")
        if key not in valid_params:
            raise ValueError(f"`{key}` is not a valid parameter name for {type(instance).__name__}.")

        if delim:
            nested_params[key][sub_key] = value
        else:
            setattr(instance, key, value)
            valid_params[key] = value

    for key, sub_params in nested_params.items():
        valid_params[key].set_params(**sub_params)
    return instance


def get_param_names(cls: Type[_BaseTpcpObject]) -> List[str]:
    """Get parameter names for the object.

    The parameters of an algorithm/pipeline are defined based on its `__init__` method.
    All parameters of this method are considered parameters of the algorithm.

    Notes
    -----
    Adopted based on :meth:`sklearn.base.BaseEstimator._get_param_names`.

    Returns
    -------
    param_names
        List of parameter names of the algorithm

    """
    parameters = list(_get_init_defaults(cls).values())
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError(
                "tpcp algorithms and pipelines should always specify their parameters in the signature of their "
                f"__init__ (no varargs). {cls} doesn't follow this convention."
            )
    # Extract and sort argument names excluding 'self'
    return sorted([p.name for p in parameters])


def _get_annotated_fields_of_type(
    instance_or_cls: Union[Type[BaseTpcpObject], BaseTpcpObject], field_type: Union[_ParaTypes, Iterable[_ParaTypes]]
) -> List[str]:
    if isinstance(field_type, _ParaTypes):
        field_type = [field_type]
    return [k for k, v in instance_or_cls.__field_annotations__.items() if v in field_type]


def _has_dangerous_mutable_default(fields: Dict[str, inspect.Parameter], cls: Type[_BaseTpcpObject]) -> None:
    mutable_defaults = []

    for name, field in fields.items():
        if _is_dangerous_mutable(field):
            # We do not raise an error right here, but wait until we checked everything to provide a more helpful
            # error message.
            mutable_defaults.append(name)

    if len(mutable_defaults) > 0:
        raise MutableDefaultsError(
            f"The class {cls.__name__} contains mutable objects as default values ({mutable_defaults}). "
            "This can lead to unexpected and unpleasant issues! "
            "To solve this the default value should be generated by a factory that produces a new value for each "
            "instance. "
            "If you simply want a new version of the provided default, wrap the value into `tpcp.CloneFactory`"
            "This will enforce a clone/copy of the respective input, whenever a new instance of your Algorithm "
            "or Pipeline is created. "
            "\n"
            "Note, that we do not check for all cases of mutable objects. "
            f"At the moment, we check only for {_get_dangerous_mutable_types()}. "
            "To learn more about this topic, check TODO: LINK."
        )


def _annotations_are_valid(fields: Dict[str, inspect.Parameter], cls: Type[_BaseTpcpObject]) -> None:
    for k, v in cls.__field_annotations__.items():
        if "__" in k:
            if v is _ParaTypes.SIMPLE:
                warnings.warn(
                    "Annotating a nested parameter (parameter like `nested_object__nest_para` as a simple "
                    "Parameter has no effect and the entire line should be removed.",
                    PotentialUserErrorWarning,
                )
        elif k not in fields:
            raise ValueError(
                f"The field '{k}' of {cls.__name__} was annotated as a `tpcp` (Hyper/Pure/Normal/Optimizable)-"
                f"Parameter, but is not a parameter listed in the init! "
                "Add the parameter to the init if it is an actual parameter of your algorithm, or remove the "
                "annotation.\n\n"
                "If you are seeing this issue when you are using `dataclasses`, this could also be a false positive."
                "Make sure you set `dataclass=True` during class definition:\n\n"
                ">>> @dataclasses.dataclass\n"
                "... class MyClass(BaseTpcpObject, dataclass=True):  # Yes, this valid Python!\n"
                "...    ...\n\n"
                ""
            )


def _has_invalid_name(fields: Dict[str, inspect.Parameter], cls: Type[_BaseTpcpObject]) -> None:
    invalid_names = [f for f in fields if "__" in f]
    if len(invalid_names) > 0:
        raise ValidationError(
            f"The parameters {invalid_names} of {cls.__name__} have a double-underscore in their name. "
            "This is not allowed, as it interferes with the nested naming conventions in tpcp.\n\n"
            "If you are seeing this while using `dataclasses`, and trying to annotate nested parameters, make sure to "
            "exclude from the init using the explicit field syntax or by wrapping it with `ClassVar`:\n\n"
            ">>> @dataclasses.dataclass\n"
            "... class MyClass(BaseTpcpObject, dataclass=True):  # Yes, this valid Python!\n"
            "...    nested__parameter: OptiPara[int] = dataclasses.field(init=False)\n"
            "...    other_nested__parameter: ClassVar[OptiPara[int]]\n\n"
        )


def _get_dangerous_mutable_types() -> Tuple[type, ...]:
    # TODO: Update this list or even make it a white list?
    return _BaseTpcpObject, list, dict, np.ndarray, pd.DataFrame, BaseEstimator


def _is_dangerous_mutable(field: inspect.Parameter) -> bool:
    """Check if a parameter is one of the mutable objects "considered" dangerous."""
    if field.default is inspect.Parameter.empty or isinstance(field.default, BaseFactory):
        return False
    if isinstance(field.default, _get_dangerous_mutable_types()):
        return True
    return False


def _is_builtin_class_instance(obj: Any) -> bool:
    return type(obj).__module__ == "builtins"


def clone(algorithm: T, *, safe: bool = False) -> T:
    """Construct a new algorithm object with the same parameters.

    This is a modified version from sklearn and the original was published under a BSD-3 license and the original file
    can be found here: https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/base.py#L31

    The method creates a copy of tpcp algorithms and pipelines, without any results attached.
    I.e. it is equivalent to creating a new instance with the same parameter.
    For all objects that are not tpcp objects (or lists of them), a deepcopy is created.

    .. warning :: This function will not clone sklearn models as expected!
                  `sklearn.clone` will remove the trained model from sklearn object by creating a new instance.
                  This clone method, will deepcopy sklearn models.
                  This means fitted models will be copied and are still available afterwards.
                  For more information have a look at the documentation about "inputs and results" in tpcp.
                  TODO: Link

    Parameters
    ----------
    algorithm : {list, tuple, set} of algorithm instance or a single algorithm instance
        The algorithm or group of algorithms to be cloned.
    safe : bool, default=False
        If safe is False, clone will fall back to a deep copy on objects that are not algorithms.

    """
    if algorithm is NOTHING:
        return algorithm
    # Handle named tuple
    if isinstance(algorithm, tuple) and hasattr(algorithm, "_asdict") and hasattr(algorithm, "_fields"):
        return type(algorithm)(*(clone(a, safe=safe) for a in algorithm))  # noqa: to-many-function-args
    # XXX: not handling dictionaries
    if isinstance(algorithm, (list, tuple, set, frozenset)):
        return type(algorithm)([clone(a, safe=safe) for a in algorithm])  # noqa: to-many-function-args
    # Compared to sklearn, we check specifically for _BaseSerializable and not just if `get_params` is defined on the
    # object.
    # Due to the way algorithms/pipelines in tpcp work, they need to inherit from _BaseSerializable.
    # Therefore, we check explicitly for that, as we do not want to accidentally treat a sklearn algo (or similar) as
    # algorithm
    if not isinstance(algorithm, BaseTpcpObject):
        if not safe:
            return copy.deepcopy(algorithm)
        raise TypeError(
            f"Cannot clone object '{repr(algorithm)}' (type {type(algorithm)}): "
            "it does not seem to be a compatible algorithm/pipline class or general `tpcp` object as it does not "
            "inherit from `BaseTpcpObject` or `Algorithm` or `Pipeline`."
        )

    klass = algorithm.__class__
    new_object_params = algorithm.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                f"Cannot clone object {algorithm}, as the constructor either does not set or modifies parameter {name}"
            )
    return new_object


class BaseFactory:
    """Baseclass for factories to circumvent mutable defaults."""

    def get_value(self) -> Any:
        """Provide the value generated by the factory.

        This method will be called every time a new instance of a class is created.
        """
        raise NotImplementedError()


class CloneFactory(BaseFactory, Generic[T]):
    """Init factory that creates a clone of the provided default values on instance initialisation.

    This can be used to make sure that each instance get their own version own copy of a mutable default of a class
    init.
    Under the hood this uses :func:`~tpcp.clone`.
    """

    def __init__(self, default_value: T):
        self.default_value = default_value

    def get_value(self) -> T:
        """Clone the default value for each instance."""
        return clone(self.default_value)


# The typing here is obviously wrong and a hack to make object that are wrapped by clone factory seem to be just of
# the same type.
# Ideally clone factory would be some generic type of proxy, but it isn't.
# Therefore, we fake the return type, so that external type checking in user code works without any errors.
# As long as `cf` is used for its intended purpose that should not matter much.
def cf(default_value: T) -> T:  # noqa: invalid-name
    """Wrap mutable default value with the `CloneFactory`.

    This is basically an alias for :class:`~tpcp.CloneFactory`
    """
    return CloneFactory(default_value)  # type: ignore
