"""Exceptions and warnings."""


class PotentialUserErrorWarning(UserWarning):
    """A warning indicating that the user might not use certain features correctly."""


class MutableDefaultsError(Exception):
    """An exception raised whenever, issues because of mutable default parameters occur/are expected."""
