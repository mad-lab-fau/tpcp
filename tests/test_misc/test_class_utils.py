import inspect

import pytest

from tpcp import BaseTpcpObject
from tpcp.misc import classproperty, set_defaults


class TestClassproperty:
    def test_simple_case(self):
        class MyClass:
            @classproperty
            def test(cls):
                return 1

        assert MyClass.test == 1


class TestSetDefaults:
    def test_func(self):
        @set_defaults(a=1, b=2)
        def function(a, b, c=3):
            return a, b, c

        assert function() == (1, 2, 3)

    def test_provided_kwargs_correctly_overwrite_defaults(self):
        @set_defaults(a=1, b=2)
        def function(a, b, c=3):
            return a, b, c

        assert function(b=5, c=4) == (1, 5, 4)

    def test_provided_args_correctly_overwrite_defaults(self):
        @set_defaults(a=1, b=2)
        def function(a, b, c=3):
            return a, b, c

        assert function(4, 5) == (4, 5, 3)

    def test_raise_with_pos_arg_missing(self):
        @set_defaults(b=2)
        def function(a, b, c=3):
            return a, b, c

        with pytest.raises(TypeError):
            assert function() == (4, 5, 3)

    def test_set_defaults_wrong_order(self):
        with pytest.raises(ValueError) as e:

            @set_defaults(a=1)
            def function(a, b, c=3):
                return a, b, c

        assert " a " in str(e.value)
        assert " b " in str(e.value)

    def test_func_with_kw_only(self):
        @set_defaults(a=1, b=2)
        def function(a, b, *, c=3):
            return a, b, c

        assert function() == (1, 2, 3)

    def test_func_with_kw_only_and_args(self):
        @set_defaults(a=1, b=2)
        def function(a, b, *args, c=3):
            return a, b, c

        assert function() == (1, 2, 3)

    def test_pos_only_args(self):
        @set_defaults(b=2)
        def function(a, b, /, c=3):
            return a, b, c

        assert function(4) == (4, 2, 3)

    def test_help_text_contains_new_defaults(self):
        @set_defaults(a=1, b=2)
        def function(a, b, c=3):
            """This is a test function"""
            return a, b, c

        help_text = str(inspect.signature(function))

        assert "a=1" in help_text
        assert "b=2" in help_text
        assert "c=3" in help_text

    def test_raises_when_overwriting_existing_defaults(self):
        with pytest.raises(ValueError):

            @set_defaults(a=1, b=2)
            def function(a, b=3):
                return a, b

    def test_works_with_class_init(self):
        class MyClass(BaseTpcpObject):
            @set_defaults(a=1, b=2)
            def __init__(self, a, b, c=3):
                self.a = a
                self.b = b
                self.c = c

        assert MyClass(4).get_params() == {"a": 4, "b": 2, "c": 3}

    def test_works_with_class_init_late_bind(self):
        class MyClass(BaseTpcpObject):
            def __init__(self, a, b, c=3):
                self.a = a
                self.b = b
                self.c = c

        MyClass.__init__ = set_defaults(a=1, b=2)(MyClass.__init__)

        assert MyClass(4).get_params() == {"a": 4, "b": 2, "c": 3}
