from examples.algorithms.algorithms_qrs_detection_final import QRSDetector
from tpcp.caching import _class_level_lru_cache_key, global_ram_cache

global_ram_cache(1)(QRSDetector)

# %%

from pathlib import Path

from examples.datasets.datasets_final_ecg import ECGExampleData

# Loading the data
try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path().resolve()
data_path = HERE.parent.parent / "example_data/ecg_mit_bih_arrhythmia/data"
example_data = ECGExampleData(data_path)
ecg_data = example_data[0].data["ecg"]

# Initialize the algorithm
algorithm = QRSDetector()
algorithm = algorithm.detect(ecg_data, example_data.sampling_rate_hz)

# %%

algorithm = QRSDetector()
algorithm = algorithm.detect(ecg_data, example_data.sampling_rate_hz)

algorithm = QRSDetector()
algorithm = algorithm.detect(ecg_data, example_data.sampling_rate_hz)


print(getattr(QRSDetector, _class_level_lru_cache_key)["cached_func"].cache_info())
