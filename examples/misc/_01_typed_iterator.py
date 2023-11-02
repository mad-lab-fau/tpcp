from dataclasses import dataclass

from tpcp.misc import TypedIterator


@dataclass
class IterResultType:
    a: int
    b: int


iterator = TypedIterator(IterResultType)

data = [1, 2, 3, 4, 5]


for d, r in iterator.iterate(data):
    r.a = d * 3

print(iterator.raw_results_)
print(iterator.a_)
print(iterator.b_)
print(iterator.inputs_)
