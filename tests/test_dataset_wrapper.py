from typing import Generic, NamedTuple, Optional, TypeVar, Union

import pandas as pd
import pytest

from tpcp import Dataset, DatasetWrapperMixin


class DomainDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "patient": ["patient_1", "patient_1", "patient_2"],
                "recording": ["recording_1", "recording_2", "recording_1"],
            }
        )

    @property
    def recordings(self) -> tuple[str, ...]:
        self.assert_is_single_group("recordings")
        return tuple(self.index["recording"])


class VariantDataset(DatasetWrapperMixin[DomainDataset], DomainDataset):
    _wrapper_groupby_cols = ("variant",)

    def __init__(
        self,
        wrapped_dataset: DomainDataset,
        variants: tuple[int, ...] = (0, 1),
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.wrapped_dataset = wrapped_dataset
        self.variants = variants
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _create_wrapped_index(self, source_index: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [source_index.assign(variant=variant) for variant in self.variants],
            ignore_index=True,
        )

    @property
    def recordings(self) -> tuple[str, ...]:
        return self.wrapped_datapoint.recordings


GroupLabelT = TypeVar("GroupLabelT", bound=tuple[str, ...])


class GenericDomainDataset(Dataset[GroupLabelT], Generic[GroupLabelT]):
    pass


class SourceGroupLabel(NamedTuple):
    patient: str
    recording: str


class VariantGroupLabel(NamedTuple):
    patient: str
    recording: str
    variant: str


class NamedSourceDataset(GenericDomainDataset[SourceGroupLabel]):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"patient": ["patient_1"], "recording": ["recording_1"]})


class NamedVariantDataset(
    DatasetWrapperMixin[NamedSourceDataset],
    GenericDomainDataset[VariantGroupLabel],
):
    _wrapper_groupby_cols = ("variant",)

    def __init__(
        self,
        wrapped_dataset: NamedSourceDataset,
        *,
        groupby_cols: Optional[Union[list[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ) -> None:
        self.wrapped_dataset = wrapped_dataset
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def _create_wrapped_index(self, source_index: pd.DataFrame) -> pd.DataFrame:
        return source_index.assign(variant="0")


class TestDatasetWrapperMixin:
    def test_links_grouping_and_resolves_wrapped_datapoint(self):
        source = DomainDataset(groupby_cols="patient")
        wrapped = VariantDataset(source)

        assert isinstance(wrapped, DomainDataset)
        assert wrapped.groupby_cols is None
        assert len(wrapped) == 4
        assert wrapped.groupby_cols == ["patient", "variant"]
        assert wrapped[0].recordings == ("recording_1", "recording_2")
        assert wrapped[0].wrapped_datapoint.groupby_cols == ["patient"]

    def test_mixin_must_precede_dataset_base(self):
        with pytest.raises(TypeError, match="DatasetWrapperMixin.*before.*Dataset"):

            class InvalidWrapper(DomainDataset, DatasetWrapperMixin[DomainDataset]):
                pass

    def test_resolves_wrapped_subset_when_grouped_only_by_wrapper_column(self):
        source = DomainDataset()
        wrapped_variant = VariantDataset(source).groupby("variant")[0]

        wrapped_subset = wrapped_variant.wrapped_datapoint

        assert wrapped_subset.groupby_cols is None
        pd.testing.assert_frame_equal(wrapped_subset.index, source.index)

    def test_wrapper_and_source_retain_their_named_group_label_types(self):
        wrapped = NamedVariantDataset(NamedSourceDataset())

        assert isinstance(wrapped.index_as_tuples()[0], VariantGroupLabel)
        assert isinstance(wrapped[0].wrapped_datapoint.index_as_tuples()[0], SourceGroupLabel)
