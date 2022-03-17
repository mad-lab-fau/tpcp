"""Download and reformat the MIT ECG database.

This script downloads a subset of the MIT-BIH Arrhythmia Database.

Note that rerunning might change the pickle files.
However, it is very likely that these changes are only caused by small differences in the compression.
These changes should not be committed to git again.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb  # noqa: import-error

selected_participants = ["100", "102", "104", "105", "106", "108", "114", "116", "119", "121", "123", "200"]

out_dir = Path(__file__).parent / "data"

for p in selected_participants:
    record = wfdb.rdsamp(p, pn_dir="mitdb")
    all_data = pd.DataFrame({"ecg": record[0][:, 0]})
    anno = wfdb.rdann(p, "atr", pn_dir="mitdb")
    all_r = np.unique(
        anno.sample[
            np.in1d(
                anno.symbol,
                ["N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"],
            )
        ]
    )
    pvc = np.unique(anno.sample[np.in1d(anno.symbol, ["V"])])
    all_r = pd.DataFrame({"R": all_r})
    pvc = pd.DataFrame({"PVC": pvc})

    all_r.to_csv(out_dir / f"{p}_all.csv")
    pvc.to_csv(out_dir / f"{p}_pvc.csv")
    all_data.to_pickle(out_dir / f"{p}.pk.gz", protocol=4)
