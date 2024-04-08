import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torchaudio as ta
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, IterableDataset


def from_datasets(
    train_dataset: Optional[Union[Dataset, Sequence[Dataset], Mapping[str, Dataset]]] = None,
    val_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    test_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    predict_dataset: Optional[Union[Dataset, Sequence[Dataset]]] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    **datamodule_kwargs: Any,
) -> "LightningDataModule":

    def dataloader(ds: Dataset, shuffle: bool = False) -> DataLoader:
        shuffle &= not isinstance(ds, IterableDataset)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def train_dataloader() -> TRAIN_DATALOADERS:
        assert train_dataset

        if isinstance(train_dataset, Mapping):
            return {key: dataloader(ds, shuffle=True) for key, ds in train_dataset.items()}
        if isinstance(train_dataset, Sequence):
            return [dataloader(ds, shuffle=True) for ds in train_dataset]
        return dataloader(train_dataset, shuffle=True)

    def val_dataloader() -> EVAL_DATALOADERS:
        assert val_dataset

        if isinstance(val_dataset, Sequence):
            return [dataloader(ds) for ds in val_dataset]
        return dataloader(val_dataset)

    def test_dataloader() -> EVAL_DATALOADERS:
        assert test_dataset

        if isinstance(test_dataset, Sequence):
            return [dataloader(ds) for ds in test_dataset]
        return dataloader(test_dataset)

    def predict_dataloader() -> EVAL_DATALOADERS:
        assert predict_dataset

        if isinstance(predict_dataset, Sequence):
            return [dataloader(ds) for ds in predict_dataset]
        return dataloader(predict_dataset)

    candidate_kwargs = {"batch_size": batch_size, "num_workers": num_workers}
    accepted_params = inspect.signature(LightningDataModule.__init__).parameters
    accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in accepted_params.values())
    if accepts_kwargs:
        special_kwargs = candidate_kwargs
    else:
        accepted_param_names = set(accepted_params)
        accepted_param_names.discard("self")
        special_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted_param_names}

    datamodule = LightningDataModule(**datamodule_kwargs, **special_kwargs)
    if train_dataset is not None:
        datamodule.train_dataloader = train_dataloader  # type: ignore[method-assign]
    if val_dataset is not None:
        datamodule.val_dataloader = val_dataloader  # type: ignore[method-assign]
    if test_dataset is not None:
        datamodule.test_dataloader = test_dataloader  # type: ignore[method-assign]
    if predict_dataset is not None:
        datamodule.predict_dataloader = predict_dataloader  # type: ignore[method-assign]

    return datamodule


class BaseSourceSeparationDataset(data.Dataset, ABC):
    def __init__(
        self,
        split: str,
        stems: List[str],
        files: List[str],
        data_path: str,
        fs: int,
        npy_memmap: bool,
        recompute_mixture: bool,
    ):
        if "mixture" not in stems:
            stems = ["mixture"] + stems

        self.split = split
        self.stems = stems
        self.stems_no_mixture = [s for s in stems if s != "mixture"]
        self.files = files
        self.data_path = data_path
        self.fs = fs
        self.npy_memmap = npy_memmap
        self.recompute_mixture = recompute_mixture

    @abstractmethod
    def get_stem(self, *, stem: str, identifier: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}
        for stem in stems:
            audio[stem] = self.get_stem(stem=stem, identifier=identifier)

        return audio

    def get_audio(self, identifier: Dict[str, Any]):
        if self.recompute_mixture:
            audio = self._get_audio(self.stems_no_mixture, identifier=identifier)
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            return self._get_audio(self.stems, identifier=identifier)

    @abstractmethod
    def get_identifier(self, index: int) -> Dict[str, Any]:
        pass

    def compute_mixture(self, audio) -> torch.Tensor:
        return sum(audio[stem] for stem in audio if stem != "mixture")
