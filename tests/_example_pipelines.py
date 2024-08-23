"""We define a couple of helper classes in a different module then the actual test to avoid issues with caching and multiprocessing."""

import warnings

from tpcp import Algorithm


class CacheWarning(UserWarning):
    pass


class ExampleClassOtherModule(Algorithm):
    _action_methods = ["action"]

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def action(self, x):
        self.result_1_ = x + self.a + self.b
        self.result_2_ = x + self.a - self.b

        # We use a warning here to make it detectable that the function was called.
        warnings.warn("This function was called without caching.", CacheWarning)
        return self
