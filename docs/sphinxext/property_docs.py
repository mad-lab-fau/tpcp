"""Prefer explicit class attribute descriptions to property getter summaries."""

import inspect
import re

from numpydoc.docscrape import NumpyDocString


def prefer_class_attribute_docs(_app, what, _name, obj, _options, lines):
    """Restore descriptions that numpydoc replaces with property summaries."""
    if what != "class" or ":Attributes:" not in lines:
        return

    attributes = NumpyDocString(inspect.getdoc(obj) or "")["Attributes"]
    if not attributes:
        return

    section_start = lines.index(":Attributes:")
    for attribute in attributes:
        if not attribute.desc or not isinstance(inspect.getattr_static(obj, attribute.name, None), property):
            continue

        # Keep numpydoc's link and type; replace only its getter-derived summary.
        heading = re.compile(rf"^    :obj:`{re.escape(attribute.name)} <[^`]+>`(?: : .*)?$")
        start = next((i for i in range(section_start + 1, len(lines)) if heading.match(lines[i])), None)
        if start is None:
            continue

        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith("        ")):
            end += 1
        lines[start + 1 : end] = [f"        {line}" for line in attribute.desc] + [""]


def setup(app):
    """Run after numpydoc has formatted the class docstring."""
    app.connect("autodoc-process-docstring", prefer_class_attribute_docs, priority=600)
    return {"parallel_read_safe": True}
