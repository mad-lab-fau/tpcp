"""Private base classes for tpcp.

These classes are in a separate module to avoid circular imports.
In basically all cases, you do not need them.
"""
from __future__ import annotations

import copy
import inspect
import warnings
from collections import defaultdict
from functools import wraps
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np

from tpcp.exceptions import MutableDefaultsError, ValidationError

Algo = TypeVar("Algo", bound="_BaseTpcpObject")


class _Nothing:
    """Sentinel class to indicate the lack of a value when ``None`` is ambiguous.

    ``_Nothing`` is a singleton. There is only ever one of it.

    This implementation is taken from the attrs package.
    """

    _singleton = None

    def __new__(cls):
        if _Nothing._singleton is None:
            _Nothing._singleton = super(_Nothing, cls).__new__(cls)
        return _Nothing._singleton

    def __repr__(self):
        return "NOTHING"

    def __bool__(self):
        return False

    def __len__(self):
        return 0  # __bool__ for Python 2


NOTHING = _Nothing()
"""
Sentinel to indicate the lack of a value when ``None`` is ambiguous.
"""


def _get_init_defaults(cls: Type[_BaseTpcpObject]) -> Dict[str, inspect.Parameter]:
    # fetch the constructor or the original constructor before deprecation wrapping if any
    init = cls.__init__
    if init is object.__init__:
        # No explicit constructor to introspect
        return {}

    # introspect the constructor arguments to find the model parameters to represent
    init_signature = inspect.signature(init)
    # Consider the constructor parameters excluding 'self'
    defaults = {k: p for k, p in init_signature.parameters.items() if p.name != "self" and p.kind != p.VAR_KEYWORD}
    return defaults


def _replace_defaults_wrapper(old_init: callable):
    """Decorate an init to create new instances of mutable defaults.

    This should only be used in combiantion with `default` and will be applied as part of `__init_subclass`.
    Direct usage of this decorator should not be required.
    """

    @wraps(old_init)
    def new_init(self, *args, **kwargs):
        # call the old init.
        old_init(self, *args, **kwargs)
        assert isinstance(self, _BaseTpcpObject)  # For the type checker
        # Check if any of the initial values has a "default parameter flag".
        # If yes we replace it with a clone (in case of a tpcp object) or a deepcopy in case of other objects.
        for k, v in self.get_params(deep=False).items():
            if isinstance(v, BaseFactory):
                setattr(self, k, v.get_value())

    return new_init


# def _collect_nested_annotations(cls: BaseTpcpObject, fields: List[attr.Attribute]):
#     normal_fields, nested_fields = gen_utils.partition(lambda x: "__" in x.name, fields)
#     cls.__nested_field_annotations__ = tuple(nested_fields)
#     for f in cls.__nested_field_annotations__:
#         if f.default != attr.NOTHING:
#             raise ValidationError(
#                 "Fields annotating nested parameters (aka fields with '__' in the name are not "
#                 "allowed to have default values!"
#                 "These fields are removed at runtime and are only there to mark nested objects as "
#                 "hyper, optimizable, or pure parameter."
#             )
#         delattr(cls, f.name)
#     return normal_fields


class _BaseTpcpObject:
    # TODO: Rething the type of nested fields
    __field_annotations__: Tuple[Tuple[str, str], ...]

    def __init_subclass__(cls, _skip_validation: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        # Because we overwrite the value for each subclass, it is the facto not inheritable
        fields = _get_init_defaults(cls)

        # Validation
        if _skip_validation is not True:
            _has_dangerous_mutable_default(fields, cls)

        if cls.__init__ is not object.__init__ and any(isinstance(field.default, BaseFactory) for field in fields.values()):
            # In case we have fields that use a factory, we need to wrap the init to replace it
            cls.__init__ = _replace_defaults_wrapper(cls.__init__)

    # TODO: Make helper api
    def _get_params_without_nested_class(self) -> Dict[str, Any]:
        return {k: v for k, v in self.get_params().items() if not isinstance(v, _BaseTpcpObject)}

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
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

    def set_params(self: Algo, **params: Any) -> Algo:
        """Set the parameters of this Algorithm.

        To set parameters of nested objects use `nested_object_name__para_name=`.
        """
        return _set_params(self, **params)

    def clone(self: Algo) -> Algo:
        """Create a new instance of the class with all parameters copied over.

        This will create a new instance of the class itself and all nested objects
        """
        return clone(self, safe=True)


class BaseTpcpObject(_BaseTpcpObject, _skip_validation=True):
    """Baseclass for all tpcp objects."""


# TODO: Make public api
def _get_params(instance: _BaseTpcpObject, deep: bool = True) -> Dict[str, Any]:
    valid_fields = get_param_names(type(instance))

    out: Dict[str, Any] = {}
    for key in valid_fields:
        value = getattr(instance, key)
        # This is a little bit of magic, that also gets the parameters of nested sklearn classifiers.
        if deep and hasattr(value, "get_params"):
            deep_items = value.get_params(deep=True).items()
            out.update((key + "__" + k, val) for k, val in deep_items)
        out[key] = value
    return out


def _set_params(instance: Algo, **params: Any) -> Algo:
    """Set the parameters of of a instance.

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


def get_annotated_fields(
    cls: Type[_BaseTpcpObject], field_type: Optional[str] = None, consider_nested_annotations: bool = False
):
    # TODO
    ...


def get_param_names(cls: Type[_BaseTpcpObject]) -> List[str]:
    """Get parameter names for the object.

    The parameters of an algorithm/pipeline are defined based on its `__init__` method.
    All parameters of this method are considered parameters of the algorithm.

    Notes
    -----
    Adopted based on `sklearn BaseEstimator._get_param_names`.

    Returns
    -------
    param_names
        List of parameter names of the algorithm

    """
    parameters = list(_get_init_defaults(cls).values())
    for p in parameters:
        if p.kind == p.VAR_POSITIONAL:
            raise RuntimeError(
                "tpcp-algorithms and pipeline should always specify their parameters in the signature of their "
                f"__init__ (no varargs). {cls} doesn't follow this convention."
            )
    # Extract and sort argument names excluding 'self'
    return sorted([p.name for p in parameters])


def _has_all_defaults(fields, cls):
    non_default_values = [f.name for f in fields if f.default is NOTHING]
    if len(non_default_values) > 0:
        raise ValidationError(
            f"The class {cls.__name__} is expected to only have arguments with a proper default "
            f"value, but for the parameters {non_default_values}, no value was provided."
        )


# def _has_only_tpcp_fields(fields, cls):
#     for f in fields:
#         if not _is_tpcp_parameter_field(f):
#             raise ValueError(
#                 "Tpcp classes are only allowed to have parameter field types that are defined by tpcp. "
#                 f"The parameter {f.name} of the class {cls.__name__} did use an incompatible field type."
#             )


def _has_dangerous_mutable_default(fields: Dict[str, inspect.Parameter], cls: Type[_BaseTpcpObject]):
    mutable_defaults = []

    for name, field in fields.items():
        if _is_dangerous_mutable(field):
            # We do not raise an error right here, but wait until we checked everything to provide a more helpfull
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


def _get_dangerous_mutable_types() -> Tuple[type, ...]:
    # TODO: Update this list or even make it a white list?
    return (_BaseTpcpObject, list, dict)


def _is_dangerous_mutable(field: inspect.Parameter):
    """Check if a parameter is one of the mutable objects "considered" dangerous."""
    # TODO: NOTHING is not the correct check here anymore
    if field.default is NOTHING or isinstance(field.default, BaseFactory):
        return False
    if isinstance(field.default, _get_dangerous_mutable_types()):
        return True
    # We check for built ins after the mutable types, as some builtins are mutables from the list
    if _is_builtin_class_instance(field.default) or field.default in (np.nan,):
        return False
    warnings.warn("Custom Object type is used! It can not be determined if it is mutable default.", UserWarning)
    return False


def _is_builtin_class_instance(obj):
    return type(obj).__module__ == "builtins"


def clone(
    algorithm: Union[BaseTpcpObject, List[BaseTpcpObject], Set[BaseTpcpObject], Tuple[BaseTpcpObject]],
    *,
    safe: bool = False,
):
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
        return NOTHING
    # XXX: not handling dictionaries
    if isinstance(algorithm, (list, tuple, set, frozenset)):
        return type(algorithm)([clone(a, safe=safe) for a in algorithm])  # noqa: to-many-function-args
    # Compared to sklearn, we check specifically for _BaseSerializable and not just if `get_params` is defined on the
    # object.
    # Due to the way algorithms/pipelines in tpcp work, they need to inherit from _BaseSerializable.
    # Therefore, we check explicitly for that, as we do not want to accidentally treat an sklearn algo (or similar) as
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

    def get_value(self):
        raise NotImplementedError()


class CloneFactory(BaseFactory):
    """Init factory that creates a clone of the provided default values on instance initialisation.

    This can be used to make sure that each instance get there own version own copy of a mutable default of a class
    init.
    Under the hood this uses :func:`~tpcp.clone`.
    """

    def __init__(self, default_value):  # noqa: super-init-not-called
        self.default_value = default_value
        self.takes_self = False

    def get_value(self):
        """Clone the default value for each instance."""
        return clone(self.default_value)


def cf(default_value: Any) -> CloneFactory:  # noqa: invalid-name
    """Wrap mutable default value with the `CloneFactory`.

    This is basically an alias for :class:`~tpcp.CloneFactory`
    """
    return CloneFactory(default_value)
