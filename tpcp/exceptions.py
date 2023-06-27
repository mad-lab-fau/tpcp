"""Exceptions and warnings."""


class PotentialUserErrorWarning(UserWarning):
    """A warning indicating that the user might not use certain features correctly."""


class ScorerFailedError(Exception):
    """An error indicating that a scorer failed."""


class OptimizationError(Exception):
    """An error indicating that the optimization of an algorithm failed."""


class TestError(Exception):
    """An error indicating that an error occurred on the testset failed."""


class ValidationError(Exception):
    """An error indicating that data-object does not comply with the guidelines."""


class MutableDefaultsError(ValidationError):
    """An exception raised whenever, issues because of mutable default parameters occur/are expected."""
