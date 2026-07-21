r"""
Transforming training datasets
==============================

Some optimization workflows need a different *set of datapoints* during training without changing the interface
expected by the pipeline. Data augmentation can multiply datapoints, while a quick smoke test might train on only a
small subset. Both operations must happen after a cross-validation fold is split so related augmented data can never
leak into the test set.

This example demonstrates both cases using ``train_dataset_transform``. A transform is simply a callable receiving a
dataset and returning a dataset that implements the same interface. The pipeline remains generic over one common
dataset type and can handle original and transformed datapoints identically.
"""

# %%
# A shared image-dataset interface
# --------------------------------
# The raw and augmented datasets use different concrete classes, but implement the same properties defined by
# ``ImageDataset``. The augmented dataset wraps the original dataset instead of copying its data. It proxies the
# label and source image through that wrapped dataset, applying only the rotation itself. Consequently, an augmented
# datapoint can be passed anywhere an original datapoint can be passed.
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tpcp import Dataset


class ImageDataset(Dataset):
    """Common interface for raw and augmented image datasets."""

    @property
    def image(self) -> np.ndarray:
        """Load the image represented by a single datapoint."""
        raise NotImplementedError

    @property
    def label(self) -> int:
        """Return the classification label of a single datapoint."""
        raise NotImplementedError


class RawImageDataset(ImageDataset):
    """Dataset containing the original images and labels."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        """Create one datapoint per source image."""
        return pd.DataFrame(
            [
                (sample_id, int(label))
                for sample_id, label in enumerate(self.labels)
            ],
            columns=["sample_id", "label"],
        )

    @property
    def image(self) -> np.ndarray:
        """Load a single original image."""
        self.assert_is_single(None, "image")
        row = self.index.iloc[0]
        return self.images[int(row["sample_id"])]

    @property
    def label(self) -> int:
        """Return the original image's classification label."""
        self.assert_is_single(None, "label")
        return int(self.index.iloc[0]["label"])


class AugmentedImageDataset(ImageDataset):
    """Dataset wrapping an image dataset and exposing lazily rotated datapoints."""

    def __init__(
        self,
        original_dataset: ImageDataset,
        rotation_degrees: tuple[int, ...] = (0, 90, 180, 270),
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.original_dataset = original_dataset
        self.rotation_degrees = rotation_degrees
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        """Add an augmentation dimension to the wrapped dataset's index."""
        return (
            pd.concat(
                [
                    self.original_dataset.index.assign(rotation_deg=rotation)
                    for rotation in self.rotation_degrees
                ],
                ignore_index=True,
            )
            .sort_values([*self.original_dataset.index.columns, "rotation_deg"])
            .reset_index(drop=True)
        )

    @property
    def _original_datapoint(self) -> ImageDataset:
        """Resolve the original datapoint represented by a single augmented row."""
        self.assert_is_single(self.groupby_cols, "_original_datapoint")
        original_index = self.index.drop(columns="rotation_deg")
        return self.original_dataset.get_subset(index=original_index)

    @property
    def image(self) -> np.ndarray:
        """Proxy the original image and apply the rotation encoded in the index."""
        self.assert_is_single(self.groupby_cols, "image")
        rotation_degrees = int(self.index.iloc[0]["rotation_deg"])
        return np.rot90(
            self._original_datapoint.image, k=rotation_degrees // 90
        ).copy()

    @property
    def label(self) -> int:
        """Proxy the label of the wrapped original datapoint."""
        return self._original_datapoint.label


# %%
# We use a tiny in-memory image dataset to keep the example fast. The first class contains variants of an L-like
# shape, while the second contains variants of a T-like shape.
images = np.asarray(
    [
        [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 1, 0]],
        [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1], [1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    ],
    dtype=float,
)
labels = np.asarray([0, 0, 0, 1, 1, 1])
raw_dataset = RawImageDataset(images, labels)


# %%
# Real augmentation: rotate images
# --------------------------------
# The transform receives only the training subset created for a fold. The augmented dataset stores that subset as its
# ``original_dataset`` parameter and builds its expanded index from it. Data and reference properties are resolved
# through the wrapped subset only when the pipeline accesses an individual datapoint.
def rotate_training_images(dataset: ImageDataset) -> AugmentedImageDataset:
    """Add 90, 180, and 270 degree rotations of every provided datapoint."""
    if dataset.groupby_cols is None:
        augmented_groupby_cols = None
    else:
        original_groupby_cols = (
            [dataset.groupby_cols]
            if isinstance(dataset.groupby_cols, str)
            else dataset.groupby_cols
        )
        augmented_groupby_cols = [*original_groupby_cols, "rotation_deg"]
    return AugmentedImageDataset(dataset, groupby_cols=augmented_groupby_cols)


augmented_dataset = rotate_training_images(raw_dataset)

# %%
# Accessing a rotated datapoint uses exactly the same ``image`` and ``label`` properties as accessing an original.
rotations = [
    augmented_dataset.get_subset(
        index=pd.DataFrame(
            augmented_dataset.index.loc[
                (augmented_dataset.index["sample_id"] == 0)
                & (augmented_dataset.index["rotation_deg"] == angle)
            ]
        )
    )
    for angle in (0, 90, 180, 270)
]
fig, axes = plt.subplots(1, 4, figsize=(7, 2))
for datapoint, angle, axis in zip(rotations, (0, 90, 180, 270), axes):
    axis.imshow(datapoint.image, cmap="gray", vmin=0, vmax=1)
    axis.set_title(f"{angle}°")
    axis.axis("off")
fig.tight_layout()

# %%
# A pipeline that handles both dataset classes
# ---------------------------------------------
# This deliberately small nearest-neighbour classifier stores the images it receives during optimization. Its only
# dataset type is ``ImageDataset``; it has no augmentation-specific logic.
from tpcp import OptimizableParameter, OptimizablePipeline


class ImageClassificationPipeline(OptimizablePipeline[ImageDataset]):
    """A minimal nearest-neighbour image classifier."""

    training_images: OptimizableParameter[Optional[np.ndarray]]
    training_labels: OptimizableParameter[Optional[np.ndarray]]

    prediction_: int

    def __init__(
        self,
        training_images: Optional[np.ndarray] = None,
        training_labels: Optional[np.ndarray] = None,
    ) -> None:
        self.training_images = training_images
        self.training_labels = training_labels

    def self_optimize(self, dataset: ImageDataset, **_: object):
        """Remember all provided images as nearest-neighbour training samples."""
        self.training_images = np.stack(
            [datapoint.image for datapoint in dataset]
        )
        self.training_labels = np.asarray(
            [datapoint.label for datapoint in dataset]
        )
        return self

    def run(self, datapoint: ImageDataset):
        """Classify an original or augmented datapoint."""
        if self.training_images is None or self.training_labels is None:
            raise RuntimeError(
                "The pipeline must be optimized before it can classify images."
            )
        distances = np.linalg.norm(
            self.training_images - datapoint.image, axis=(1, 2)
        )
        self.prediction_ = int(self.training_labels[np.argmin(distances)])
        return self


def accuracy(
    pipeline: ImageClassificationPipeline, datapoint: ImageDataset
) -> float:
    """Score one image."""
    return float(pipeline.safe_run(datapoint).prediction_ == datapoint.label)


# %%
# Fold-local augmentation
# -----------------------
# ``cross_validate`` first creates the untouched train and test subsets. ``Optimize`` then transforms only the train
# subset immediately before ``self_optimize``. The scorer therefore continues to evaluate original images.
from sklearn.model_selection import StratifiedKFold
from tpcp.optimize import Optimize
from tpcp.validate import DatasetSplitter, cross_validate

cv = DatasetSplitter(StratifiedKFold(n_splits=3), stratify="label")

augmentation_results = cross_validate(
    Optimize(
        ImageClassificationPipeline(),
        train_dataset_transform=rotate_training_images,
    ),
    raw_dataset,
    scoring=accuracy,
    cv=cv,
    return_optimizer=True,
    progress_bar=False,
)
augmentation_training_sizes = [
    len(optimizer.transformed_dataset_)
    for optimizer in augmentation_results["optimizer"]
]
print(augmentation_training_sizes)


# %%
# Subsampling for a smoke test
# ----------------------------
# A smoke-test transform follows the same interface. Here we keep one training datapoint per class, reducing every
# fold from four training images to two while leaving its two scored test images untouched.
def subsample_for_smoke_test(dataset: ImageDataset) -> ImageDataset:
    """Keep one datapoint per class for a fast end-to-end check."""
    smoke_test_index = dataset.index.groupby(
        "label", sort=True, as_index=False
    ).head(1)
    return dataset.get_subset(index=smoke_test_index)


smoke_test_results = cross_validate(
    Optimize(
        ImageClassificationPipeline(),
        train_dataset_transform=subsample_for_smoke_test,
    ),
    raw_dataset,
    scoring=accuracy,
    cv=cv,
    return_optimizer=True,
    progress_bar=False,
)
smoke_test_training_sizes = [
    len(optimizer.transformed_dataset_)
    for optimizer in smoke_test_results["optimizer"]
]
print(smoke_test_training_sizes)

# %%
# The important distinction is visible in both result sets: the optimizer sees the transformed training dataset, but
# ``test__data_labels`` and all reported test scores still refer to the original datapoints.
