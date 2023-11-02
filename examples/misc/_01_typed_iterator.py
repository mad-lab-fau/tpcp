from dataclasses import dataclass

import pandas as pd

from tpcp.misc import TypedIterator


@dataclass
class IterResultType:
    a: int
    b: pd.DataFrame


aggregations = [("b", lambda i, r: pd.concat(r, keys=i))]


iterator = TypedIterator(IterResultType, aggregations=aggregations)

data = [1, 2, 3, 4, 5]


for d, r in iterator.iterate(data):
    r.a = d * 3
    r.b = pd.DataFrame({"v1": d * 1, "v2": d * 2, "v3": d * 3, "v4": d * 4, "v5": d * 5}, index=[0])

print(iterator.raw_results_)
print(iterator.a_)
print(iterator.b_)
print(iterator.inputs_)
