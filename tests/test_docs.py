"""Checks for rendered API documentation."""

from pathlib import Path
from textwrap import dedent

from sphinx.application import Sphinx


def test_class_attribute_description_takes_precedence_over_property_docstring(tmp_path):
    """A class attribute description wins without hiding its type or property details."""
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    (tmp_path / "conf.py").write_text(
        f"import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        f"sys.path.insert(0, {str(docs_dir)!r})\n"
        "extensions = ['sphinx.ext.autodoc', 'numpydoc', 'sphinxext.property_docs']\n"
    )
    (tmp_path / "fixture.py").write_text(
        dedent('''\
            class Example:
                """Example class.

                Attributes
                ----------
                value : int
                    Class attribute description.
                    Another sentence.
                """

                @property
                def value(self) -> int:
                    """Property getter description."""
                    return 1
            ''')
    )
    (tmp_path / "index.rst").write_text("Example\n=======\n\n.. autoclass:: fixture.Example\n   :members:\n")

    app = Sphinx(
        srcdir=tmp_path,
        confdir=tmp_path,
        outdir=tmp_path / "out",
        doctreedir=tmp_path / "doctrees",
        buildername="text",
        warningiserror=True,
        freshenv=True,
    )
    app.build()

    rendered = (tmp_path / "out" / "index.txt").read_text()
    attributes, property_detail = rendered.split("property value: int")
    assert '"value" : int' in attributes
    assert "Class attribute description. Another sentence." in attributes
    assert "Property getter description." not in attributes
    assert "Property getter description." in property_detail
