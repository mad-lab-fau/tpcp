"""A custom hash function implementation that properly supports pytorch."""
import io
import pickle
import struct
import sys

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

    def hash(self, obj, return_digest=True):  # noqa: A003
        """Get hash while handling some edgecases.

        Namely, this implementation fixes the following issues:

        - Because we skip memoization, we need to handle the case where the object is self-referential.
          We just catch the error and raise a more descriptive error message.


        """
        try:
            return super().hash(obj, return_digest)
        except RecursionError as e:
            raise ValueError(
                "The custom hasher used in tpcp does not support hashing of self-referential objects."
            ) from e

    def save_global(self, obj, name=None, pack=struct.pack):
        """We overwrite this method to fix the issue with objects defined in the `__main__` module.

        We need to handle the case where the object is defined in the `__main__` module.
        For some reason, this can lead to pickle issues.
        Based on some information I found, this should not happen, but it still does...
        To fix it, we detect, when an object is defined in `__main__` and temporarily add it to the "real" module
        representing the main function.
        Afterwards we do some cleanup.
        Not sure if really required, but it seems to work.
        Overall very hacky, but I don't see a better way to fix this.

        The implementation provided by the parent class (joblib.hashing.Hasher) should work, but it does not.
        I think the second call to `save_global` there is causing the issue.
        In our version we first fix the module before calling save_global again.

        Note that we are not using super() here, as we want to skip the faulty implementation in the parent class.

        We also need to modify the dispatch table to force our custom save_global method.
        See below this method for the implementation.
        """
        kwargs = {"name": name, "pack": pack}
        del kwargs["pack"]
        try:
            Pickler.save_global(self, obj, **kwargs)
        except pickle.PicklingError:
            modules_modified = []
            if getattr(obj, "__module__", None) == "__main__":
                try:
                    name = obj.__qualname__
                    to_add_obj = obj
                except AttributeError:
                    name = obj.__class__.__qualname__
                    to_add_obj = obj.__class__
                mod = sys.modules["__main__"]
                if not hasattr(mod, name):
                    modules_modified.append((mod, name))
                    setattr(mod, name, to_add_obj)
            try:
                Pickler.save_global(self, obj, **kwargs)
            finally:
                # Remove all new entries made to the main module.
                for mod, name in modules_modified:
                    delattr(mod, name)

    # We also need to rewrite the dispatch table to force our custom save_global method.
    dispatch = Hasher.dispatch.copy()
    # builtin
    dispatch[type(len)] = save_global
    # type
    dispatch[type(object)] = save_global
    # classobj
    dispatch[type(Hasher)] = save_global
    # function
    dispatch[type(pickle.dump)] = save_global


class NoMemoizeNumpyHasher(NoMemoizeHasher, NumpyHasher):
    """A joblib numpy hasher with all memoize features disabled."""


class TorchHasher(NoMemoizeNumpyHasher):
    """A hasher that can handle torch models.

    Under the hood this uses the new implementation of `torch.save` to serialize the model.
    This produces consistent output.

    Note: I never did any performance checks with large models.
    """

    def __init__(self, hash_name="md5", coerce_mmap=False):
        super().__init__(hash_name, coerce_mmap)
        import torch  # pylint: disable=import-outside-toplevel

        self.torch = torch

    def save(self, obj):
        if isinstance(obj, (self.torch.nn.Module, self.torch.Tensor)):
            b = b""
            buffer = io.BytesIO(b)
            self.torch.save(obj, buffer)
            self._hash.update(b)
            return
        NumpyHasher.save(self, obj)


# This function is modified based on
# https://github.com/joblib/joblib/blob/4dafaff788a3b5402acfed091558b4c511982959/joblib/hashing.py#L244
def custom_hash(obj, hash_name="md5", coerce_mmap=False):
    """Quick calculation of a hash to identify uniquely Python objects containing numpy arrays and torch models.

    This function is modified based on `joblib.hash` so that it can properly handle torch models.

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
    if "torch" in sys.modules:
        hasher = TorchHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    elif "numpy" in sys.modules:
        hasher = NoMemoizeNumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = NoMemoizeHasher(hash_name=hash_name)
    return hasher.hash(obj)
