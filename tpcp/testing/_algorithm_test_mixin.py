"""A mixin for all common tests that should be run on all algorithm classes."""
import inspect

import joblib
import pytest
from numpydoc.docscrape import NumpyDocString

from tpcp import (
    Algorithm,
    BaseFactory,
    get_action_method,
    get_action_params,
    get_param_names,
    get_results,
    make_action_safe,
)
from tpcp._base import BaseTpcpObjectT, _BaseTpcpObject
from tpcp._hash import custom_hash


class TestAlgorithmMixin:
    """A mixin for all common tests that should be run on all algorithm classes.

    You can use this mixin to test your algorithm class OR Pipeline sub-class (as pipelines are just algorithms).

    This mixin is intended to be used in a pytest testclass.
    For this, create a new class that inherits from this mixin and set the ALGORITHM_CLASS attribute to the algorithm
    class that should be tested.
    Then set `__test__ = True` to enable the tests.

    You can further customize the tests by setting the following attributes:

    - ONLY_DEFAULT_PARAMS: If set to False, the testclass can have non-optional kwargs in the constructor.
    - ASSUME_HASHABLE: If set to False, the testclass is not expected to be hashable by joblib.
    - CHECK_DOCSTRING: If set to False, the docstring tests are skipped.

    In some very specific cases you might want to ignore some parameters in the docstring tests.
    For this, set the _IGNORED_NAMES attribute to a tuple of parameter names that should be ignored.

    Examples
    --------
    >>> class TestMyAlgorithm(TestAlgorithmMixin):
    ...    ALGORITHM_CLASS = MyAlgorithm
    ...    __test__ = True
    ...
    ...    ONLY_DEFAULT_PARAMS = False

    """

    ALGORITHM_CLASS: type[Algorithm]
    __test__ = False

    ONLY_DEFAULT_PARAMS: bool = True
    ASSUME_HASHABLE: bool = True
    CHECK_DOCSTRING: bool = True

    _IGNORED_NAMES = ()

    @pytest.fixture()
    def after_action_instance(self, **kwargs) -> BaseTpcpObjectT:  # noqa: PT004
        """Return an instance of the algorithm class AFTER the action is performed.

        This needs to be implemented by every testclass.
        The returned algorithm instance should have the result attributes and the "other parameters" (i.e. the action
        method inputs) set.
        """
        raise NotImplementedError

    def test_is_algorithm(self):
        """Test that the class is actually an algorithm."""
        assert issubclass(self.ALGORITHM_CLASS, Algorithm)

    def test_after_action_instance_valid(self, after_action_instance):
        """Test that the implemented after_action_fixture returns a valid instance."""
        assert isinstance(after_action_instance, Algorithm)
        assert isinstance(after_action_instance, self.ALGORITHM_CLASS)

    def test_init(self):
        """Test that all init paras are passed through untouched."""
        field_names = get_param_names(self.ALGORITHM_CLASS)
        init_dict = {k: k for k in field_names}

        test_instance = self.ALGORITHM_CLASS(**init_dict)

        for k, v in init_dict.items():
            assert getattr(test_instance, k) == v, k

    def test_only_optional_kwargs(self):
        """Test that the class has only optional kwargs."""
        if self.ONLY_DEFAULT_PARAMS is False:
            pytest.skip("The testclass is not expected to have only optional kwargs.")
        self.ALGORITHM_CLASS()

    def test_algorithm_can_be_cloned(self, after_action_instance):
        """Test if an algorithm can be cloned."""
        initial = after_action_instance.clone()
        assert custom_hash(initial) == custom_hash(initial.clone())

    def test_all_parameters_documented(self):
        """Test if all init-parameters are documented."""
        if not self.CHECK_DOCSTRING:
            pytest.skip("Docstring testing explicitly disabled.")
        docs = NumpyDocString(inspect.getdoc(self.ALGORITHM_CLASS))

        documented_names = {p.name for p in docs["Parameters"]}
        documented_names -= set(self._IGNORED_NAMES)
        actual_names = set(get_param_names(self.ALGORITHM_CLASS))
        actual_names -= set(self._IGNORED_NAMES)

        assert documented_names == actual_names

    def test_all_attributes_documented(self, after_action_instance):
        """Test if all result attributes are documented."""
        if not self.CHECK_DOCSTRING:
            pytest.skip("Docstring testing explicitly disabled.")
        if not after_action_instance:
            pytest.skip("The testclass did not implement the correct `after_action_instance` fixture.")
        docs = NumpyDocString(inspect.getdoc(self.ALGORITHM_CLASS))

        documented_names = {p.name for p in docs["Attributes"]}
        documented_names -= set(self._IGNORED_NAMES)
        actual_names = set(get_results(after_action_instance).keys())
        actual_names -= set(self._IGNORED_NAMES)

        assert documented_names == actual_names

    def test_all_other_parameters_documented(self, after_action_instance):
        """Test if all other parameters (action method inputs) are documented."""
        if not self.CHECK_DOCSTRING:
            pytest.skip("Docstring testing explicitly disabled.")
        if not after_action_instance:
            pytest.skip("The testclass did not implement the correct `after_action_instance` fixture.")
        docs = NumpyDocString(inspect.getdoc(self.ALGORITHM_CLASS))

        documented_names = {p.name for p in docs["Other Parameters"]}
        documented_names -= set(self._IGNORED_NAMES)
        actual_names = set(get_action_params(after_action_instance).keys())
        actual_names -= set(self._IGNORED_NAMES)

        assert documented_names == actual_names

    def test_action_method_returns_self(self, after_action_instance):
        """Test if the action method returns self."""
        # call the action method a second time to test the output
        parameters = get_action_params(after_action_instance)
        results = get_action_method(after_action_instance)(**parameters)

        assert id(results) == id(after_action_instance)

        assert id(results) == id(after_action_instance)

    def test_set_params_valid(self, after_action_instance):
        """Test that set_params works."""
        instance = after_action_instance.clone()
        valid_names = get_param_names(instance)
        values = list(range(len(valid_names)))
        instance.set_params(**dict(zip(valid_names, values)))

        for k, v in zip(valid_names, values):
            assert getattr(instance, k) == v, k

    def test_set_params_invalid(self, after_action_instance):
        """Test that set_params raises an error if an invalid parameter is passed."""
        instance = after_action_instance.clone()

        with pytest.raises(ValueError, match="`an_invalid_name` is not a valid parameter") as e:
            instance.set_params(an_invalid_name=1)

        assert "an_invalid_name" in str(e)
        assert self.ALGORITHM_CLASS.__name__ in str(e)

    def test_hashing(self, after_action_instance):
        """Check if caching with joblib will work as expected."""
        if self.ASSUME_HASHABLE is False:
            pytest.skip("The testclass is not expected to be hashable.")
        instance = after_action_instance.clone()

        assert joblib.hash(instance) == joblib.hash(instance.clone())

    def test_nested_algo_marked_default(self):
        """Test that nested algorithms are wrapped with cf/CloneFactory."""
        init = self.ALGORITHM_CLASS.__init__
        if init is object.__init__:
            # No explicit constructor to introspect
            pytest.skip()

        # introspect the constructor arguments to find the _model parameters to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = {
            p.name: p.default
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        }
        nested_algos = {k: v for k, v in parameters.items() if isinstance(v, (_BaseTpcpObject, BaseFactory))}
        if len(nested_algos) == 0:
            pytest.skip()

        # If nested algos exists, we check that we get a new instance of the nested object and not the mutable default.
        # If not, we let the test fail, as we should always wrap such paras in a default explicitly.
        new_instance = self.ALGORITHM_CLASS().get_params()
        for k, v in nested_algos.items():
            assert new_instance[k] is not v, "nested algorithm defaults should be wrapped in `cf`/`CloneFactory`."

    def test_passes_safe_action_checks(self, after_action_instance):
        """Test that the algorithm passes the safe action checks."""
        # We just wrap the method and call it.
        # We don't care about the return value
        make_action_safe(get_action_method(after_action_instance))(
            after_action_instance,
            **get_action_params(after_action_instance),
        )
