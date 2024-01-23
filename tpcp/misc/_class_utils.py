"""Some utility functions for classes."""
import functools
import inspect
from typing import Any, Callable, Generic, Optional, TypeVar

from typing_extensions import Concatenate, ParamSpec, Self, Unpack

P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


class classproperty(Generic[P, T, R]):  # noqa: N801
    """
    A decorator that converts a class method into a class property.

    This decorator allows a class method, which takes the class as its only argument,
    to be accessed directly from the class as a property. This can be useful when
    you want to compute a value that is associated with the class, but not with
    any specific instance of the class.

    This implementation is taken from Django and slightly modified:
    https://docs.djangoproject.com/en/5.0/ref/utils/#django.utils.functional.classproperty
    The original version in Django is published under the BSD license.

    Example
    -------
    >>> class MyClass:
    ...     @classproperty
    ...     def name(cls) -> str:
    ...         return cls.__name__
    >>> print(MyClass.name)
    'MyClass'

    """

    def __init__(self, method: Optional[Callable[Concatenate[type[T], P], R]] = None):
        self.fget = method

    def __get__(self, instance: Optional[T], cls: Optional[type[T]] = None) -> R:
        """Return the class property value."""
        return self.fget(cls)

    def getter(self, method: Callable[Concatenate[type[T], P], R]) -> Self:
        """Override the decorated method."""
        self.fget = method
        return self


def set_defaults(**defaults: Unpack[dict[str, Any]]) -> Callable[[Callable[P, R]], Callable[P, R]]:  # noqa: C901
    """Set the default values of a function's parameters.

    This decorator returns a wrapped version of the function, with the specified default values set.
    This can be useful when you want to set default values programmatically, such as loading them from a file.

    This decorator also works on class methods, and can be used to set default values of class initialization parameters.

    Note that the new default values must be a subset of the function's parameters.
    Also, you cannot set default values for parameters that already have a default value.
    Doing so will raise a ValueError.

    Parameters
    ----------
    **defaults
        The new default values for the function's parameters.

    Returns
    -------
    decorator
        A decorator that can be applied to a function to modify its default values.

    Example
    -------
    >>> @set_defaults(a=10, b=20)
    ... def add(a, b):
    ...     return a + b
    >>> print(add())
    30
    >>> print(add(5))
    25
    >>> print(add(5, 5))
    10
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = inspect.signature(func)

        new_paras = [
            inspect.Parameter(name, kind, default=default)
            for name, (kind, default) in zip(
                sig.parameters,
                [(p.kind, defaults.get(p.name, p.default)) for p in sig.parameters.values()],
            )
        ]

        # Check that when adding defaults to originally positional arguments that we don't have any positional without
        # default values after positional arguments with default values.
        # This check will also be performed when replacing the signature, but we want to raise a more specific error,
        # as this will be a common error people run into.
        pos_args = [p for p in new_paras if p.kind == p.POSITIONAL_OR_KEYWORD]
        # We backwards and check for the first parameter with a default value.
        # All values after that must have a default value, otherwise we raise an error.
        first_default_found_at = None
        for p in pos_args:
            if first_default_found_at is None:
                if p.default is not inspect.Parameter.empty:
                    first_default_found_at = p
            elif p.default is inspect.Parameter.empty:
                raise ValueError(
                    "When adding default values to positional arguments, "
                    "all positional arguments after the first argument with a default value must also have a "
                    "default value. "
                    f"Currently, {first_default_found_at.name} has a default value, but {p.name} "
                    f"(which comes after {first_default_found_at.name} in the function signature) does not."
                )

        try:
            new_sig = sig.replace(parameters=new_paras)
        except ValueError as e:
            raise ValueError(
                f"Could not set defaults for {func.__name__}. " "Check the error above for more information."
            ) from e

        # Check that we don't overwrite existing default values
        for name in defaults:
            if sig.parameters[name].default is not inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter {name} already has a default value. "
                    "We only allow adding, but not changing existing default values to avoid confusion."
                )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if class method.
            # This is needed to pass through self untouched.
            # Otherwise, the process of binding the arguments might access attributes of the class instance, that are
            # not yet available.
            if inspect.ismethod(func):
                # We need to remove the first argument, as this is the class instance
                args = args[1:]
                self = [args[0]]
            else:
                self = []

            # We bind the arguments to the new signature, to replace the default values
            # This will raise a TypeError if the arguments are not compatible with the new signature
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return func(*self, *bound.args, **bound.kwargs)

        # Update function signature

        wrapper.__signature__ = new_sig

        return wrapper

    return decorator
