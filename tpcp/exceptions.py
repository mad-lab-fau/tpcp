"""Exceptions and warnings."""


class PotentialUserErrorWarning(UserWarning):
    """A warning indicating that the user might not use certain features correctly."""


class ScorerFailed(UserWarning):
    """A warning indicating that a scorer failed."""


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class MutableDefaultsError(ValidationError):
    """An exception raised whenever, issues because of mutable default parameters occur/are expected."""
