from pathlib import Path


def test_custom_dataset():
    # There is not really anything specific, we want to test here, so we just run everything and check that there are
    # no errors.
    import examples.datasets.datasets_basics  # noqa


def test_real_life_dataset():
    from examples.datasets.datasets_real_world_example import ECGExampleData

    dataset = ECGExampleData(data_path=Path("../../example_data/ecg_mit_bih_arrhythmia/data"))
    assert dataset.index.shape == (12, 2)

    # Just test that accessing them does not produce any errors
    subset = dataset[0]
    subset.data
    subset.sampling_rate_hz
    subset.labeled_r_peaks_
    subset.pvc_positions_
    subset.r_peak_positions_