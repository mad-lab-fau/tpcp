"""A custom hash function implementation that properly supports pytorch."""
import io
import sys

from joblib.hashing import Hasher, NumpyHasher


class TorchHasher(NumpyHasher):
    """A hasher that can handle torch models.

    Under the hood this uses the new implementation of `torch.save` to serialize the model.
    This produces consistent output.

    Note: I never did any performance checks with large models.
    """

    def __init__(self, hash_name="md5", coerce_mmap=False):
        super().__init__(hash_name, coerce_mmap)
        import torch  # noqa: import-outside-toplevel

        self.torch = torch

    def save(self, obj):
        if isinstance(obj, self.torch.nn.Module):
            b = bytes()
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
        hasher = NumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = Hasher(hash_name=hash_name)
    return hasher.hash(obj)
