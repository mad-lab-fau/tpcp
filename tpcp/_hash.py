"""A custom hash function implementation that properly supports pytorch."""

import contextlib
import os
import pickle
import sys
import types
import warnings
from pathlib import Path

from joblib.func_inspect import get_func_code
from joblib.hashing import Hasher, NumpyHasher

Pickler = pickle._Pickler


class NoMemoizeHasher(Hasher):
    """A joblib hasher with all memoize features disabled."""

    def memoize(self, obj):  # noqa: ARG002
        # We skip memoization here, as it can cause some issues with nested objects.
        # https://github.com/joblib/joblib/issues/1283
        # In particular with the way cloning is implemented in tpcp, such bugs might occur more often than in other
        # applications.
        # My understanding is that memoization is used to reduce the size of the output.
        # Not using memoization is actually faster, but will fail with self referential objects.
        # (https://docs.python.org/3/library/pickle.html#pickle.Pickler.fast).
        # In these cases hashing will throw an `RecursionError`.
        # So it seems like it is a tradeoff between the two issues.
        # For now, we accept the recursion issue, as I think this might happen less often by accident.
        return

    def hash(self, obj, return_digest=True):
        """Get hash while handling some edgecases.

        Namely, this implementation fixes the following issues:

        - Hashing of globally and objects defined in closures, namely dynamic classes and functions.
          We do that by not hashing the object completely, but only the qualname and the code object.
        - Because we skip memoization, we need to handle the case where the object is self-referential.
          We just catch the error and raise a more descriptive error message.


        """
        try:
            return super().hash(obj, return_digest)
        except RecursionError as e:
            raise ValueError(
                "The custom hasher used in tpcp does not support hashing of self-referential objects."
            ) from e

    def save(self, obj):
        try:
            super().save(obj)
        except pickle.PicklingError:
            if not ("<locals>" in obj.__qualname__ or "__main__" in obj.__module__):
                raise
            # These are problematic cases that sometimes lead to issues.
            # We need to handle the case where an obj is defined in a closure.
            # This can lead to issues, as the object is not hashable.
            # We therefore destruct the object into its parts and cache them individually.
            # https://stackoverflow.com/questions/46768213/how-to-hash-a-class-or-function-definition
            #
            # It might be that there are rare cases, where this is not enough,
            # and we would get false-positives equality of objects.
            # However, in the context of tpcp, that is not really a concern. In most possible cases, this just means
            # that some (likely obscure) guardrail will not trigger for you.
            if isinstance(obj, types.FunctionType):
                if "<lambda>" in obj.__qualname__:
                    warnings.warn(
                        "You are attempting to hash a lambda defined within a closure, likely because you used it as a "
                        "parameter to a tpcp object (e.g. an Aggregator). "
                        "While this works most of the time, it can to lead to some unexpected false positive hash "
                        "equalities, depending on how you define the lambdas. "
                        "We highly recommend to use a named function or a `functools.partial` instead.",
                        stacklevel=1,
                    )
                # Note, that for lambdas this actully hashes the entire definition line.
                # This means potentially more of the surrounding code than the lambda itself is hashed.
                obj = ("F", obj.__qualname__, get_func_code(obj), vars(obj))

            if isinstance(obj, type):
                obj = ("C", obj.__qualname__, obj.__bases__, dict(vars(obj)))

            super().save(obj)


class NoMemoizeNumpyHasher(NoMemoizeHasher, NumpyHasher):
    """A joblib numpy hasher with all memoize features disabled."""


class NNHasher(NoMemoizeNumpyHasher):
    """A hasher that can handle torch models.

    Under the hood this uses the new implementation of `torch.save` to serialize the model.
    This produces consistent output.

    Note: I never did any performance checks with large models.
    """

    def __init__(self, hash_name="md5", coerce_mmap=False) -> None:
        super().__init__(hash_name, coerce_mmap)
        try:
            import torch  # pylint: disable=import-outside-toplevel

            self.torch = torch
        except ImportError:
            self.torch = None
        try:
            import tensorflow

            self.tensorflow = tensorflow
        except ImportError:
            self.tensorflow = None

    def _convert_tensors_to_numpy(self, obj):
        # Recursively convert torch tensors in obj to numpy arrays
        if isinstance(obj, dict):
            for key, value in obj.items():
                obj[key] = self._convert_tensors_to_numpy(value)
        if isinstance(obj, self.torch.nn.Module):
            state_dict = obj.state_dict()
            obj = {key: self._convert_tensors_to_numpy(value) for key, value in state_dict.items()}
            return obj
        if isinstance(obj, self.torch.Tensor):
            obj_as_numpy = obj.cpu().detach().numpy()
            return obj_as_numpy
        return obj

    def save(self, obj):
        if self.tensorflow and isinstance(obj, (self.tensorflow.keras.Model,)):
            # The normal tensorflow objects don't have a consistent hash.
            # Therefore, we need to serialize all relevant information.
            # I hope by using `serialize_keras_object` we get all important parts.
            # The `SharedObjectSavingScope` is required to make sure that shared objects are properly serialized.
            from tensorflow.python.keras.utils.generic_utils import SharedObjectSavingScope, serialize_keras_object

            with SharedObjectSavingScope():
                super().save(
                    [obj.__class__.__name__, serialize_keras_object(obj), obj.get_weights()],
                )
            return

        if self.torch and isinstance(obj, (self.torch.nn.Module, self.torch.Tensor)):
            obj = self._convert_tensors_to_numpy(obj)

        super().save(obj)


# This function is modified based on
# https://github.com/joblib/joblib/blob/4dafaff788a3b5402acfed091558b4c511982959/joblib/hashing.py#L244
def custom_hash(obj, hash_name="md5", coerce_mmap=False):
    """Quick calculation of a hash to identify uniquely Python objects containing numpy arrays and torch models.

    This function is modified based on `joblib.hash` so that it can properly handle torch and tensorflow objects.
    It adds some further "fixes" for dynamically defined functions.

    Parameters
    ----------
    obj
        The object to be hashed
    hash_name: 'md5' or 'sha1'
        Hashing algorithm used. sha1 is supposedly safer, but md5 is faster.
    coerce_mmap: boolean
        Make no difference between np.memmap and np.ndarray

    """
    valid_hash_names = ("md5", "sha1")
    if hash_name not in valid_hash_names:
        raise ValueError(f"Valid options for 'hash_name' are {valid_hash_names}. Got hash_name={hash_name!r} instead.")
    if "torch" in sys.modules or "tensorflow" in sys.modules:
        hasher = NNHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    elif "numpy" in sys.modules:
        hasher = NoMemoizeNumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = NoMemoizeHasher(hash_name=hash_name)
    with Path(os.devnull).open("w") as devnull, contextlib.redirect_stdout(devnull):
        # Some object decide to print stuff to stdout when pickling.
        # As we potentially pickle a lot of objects, we don't want to see this.
        return hasher.hash(obj)
