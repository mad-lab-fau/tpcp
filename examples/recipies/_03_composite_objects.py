r"""
.. _compositeobjects:

Composite-Algorithms and Pipelines
==================================

Sometimes a pipeline or algorithms requires a list of parameters or nested objects.
As we can not support parameters which names are not known when the class is defined, such cases need to be handled
via composite fields.

A composite field is a parameter expecting a value of the shape `[(name1, sub_para1), (name2, sub_para2), ...]`.
The sub-paras can themselves be tpcp objects.

As it is difficult at runtime to know, if a parameter is expected to be a composite field, you need to actively
specify all fields that should be considered composite fields during class definition using the `_composite_params`
attribute:
"""
import dataclasses
import traceback
from typing import Optional

from tpcp import Pipeline
from tpcp.exceptions import ValidationError


@dataclasses.dataclass
class Workflow(Pipeline):
    _composite_params = ("pipelines",)

    pipelines: Optional[list[tuple[str, Pipeline]]] = None

    def __init__(self, pipelines=None):
        self.pipelines = pipelines


# %%
# That's it!
# Now tpcp knows, that `pipelines` should be a composite field and will actually complain, if we try to assign
# something invalid.
# Composite fields are allowed to either have the value None, or be a list of tuples as explained above
instance = Workflow()
instance.pipelines  # Our default value of None

# %%
instance.pipelines = "something invalid"
try:
    print(instance.get_params())
except ValidationError:
    traceback.print_exc()


# %%
# While you could set the individual sub-params in a composite field to whatever you want, the real value of explicit
# composite fields are the use of tpcp-objects
@dataclasses.dataclass
class MyPipeline(Pipeline):
    param: float = 4
    param2: int = 10


workflow_instance = Workflow(pipelines=[("pipe1", MyPipeline()), ("pipe2", MyPipeline(param2=5))])
# %%
# We can now use `get_params` to get a deep inspection of the nested objects:
workflow_instance.get_params(deep=True)

# %%
# Or we can set params using the following syntax:
workflow_instance = workflow_instance.set_params(pipelines__pipe1__param=2, pipelines__pipe2=MyPipeline(param2=4))
workflow_instance.get_params(deep=True)

# %%
# Note that it is not possible to set parameters for keys that don't exist yet!
# In such a case, you would manually recreate the full list.
