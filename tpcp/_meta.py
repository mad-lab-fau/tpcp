"""The algorithm meta-class that checks for correct implementation of Pipelines."""
from inspect import Parameter
from typing import Tuple

from tpcp import _base
from tpcp._utils._general import _DEFAULT_PARA_NAME, clone
from tpcp.exceptions import MutableDefaultsError


def _get_dangerous_mutable_types() -> Tuple[type]:
    return (_base._BaseTpcpObject,)


def _is_dangerous_mutable(para: Parameter):
    val = para.default
    if isinstance(val, _get_dangerous_mutable_types()) and not getattr(val, _DEFAULT_PARA_NAME, None):
        return True
    return False


class AlgorithmMeta(type):
    """Meta class for all algorithms.

    This does basically two things:

    Whenever you use a nested algorithm as parameter, it checks that it is wrapped in `default` to fix issues with
    mutable defaults.
    This check happens at class creation time.

    Second, when an instance of a class is created, a copy of all objects that are wrapped as `default` is created
    and used instead of the mutable default.
    To create the copy, the `tpcp.clone` method is used.
    """

    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        assert issubclass(cls, _base._BaseTpcpObject)
        init_defaults = cls._get_init_defaults()
        if init_defaults:
            dangerous_mutables = {k: v.default for k, v in init_defaults.items() if _is_dangerous_mutable(v)}
            if len(dangerous_mutables) > 0:
                raise MutableDefaultsError(
                    f"The class {name} contains mutable objects as default values ({dangerous_mutables}). "
                    "This can lead to unexpected and unpleasant issues! "
                    "To solve this issue wrap your mutable default arguments explicitly with `default` "
                    "(or `df`). "
                    "This will enforce a clone/copy of the respective input, whenever a new instance of "
                    "your Algorithm or Pipeline is created. "
                    "\n"
                    "Note, that we do not check for all cases of mutable objects. "
                    f"At the moment, we check only for {_get_dangerous_mutable_types()}. "
                    "To learn more about this topic, check TODO: LINK."
                )

    def __call__(cls, *args, **kwargs):
        # Overwriting call overwrites the instance creation of the final class
        instance = super().__call__(*args, **kwargs)
        assert isinstance(instance, _base._BaseTpcpObject)
        # Check if any of the initial values has a "default parameter flag".
        # If yes we replace it with a clone (in case of a tpcp object) or a deepcopy in case of other objects.
        for k, v in instance.get_params(deep=False).items():
            if getattr(v, _DEFAULT_PARA_NAME, None):
                cloned_object = clone(v)
                # In case we made a deepcopy, we need to remove the default param.
                try:
                    delattr(cloned_object, _DEFAULT_PARA_NAME)
                except AttributeError:
                    pass
                setattr(instance, k, cloned_object)
        return instance
