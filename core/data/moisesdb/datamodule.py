import os.path
from typing import Mapping, Optional

import pytorch_lightning as pl

from core.data.base import from_datasets
from core.data.moisesdb.dataset import MoisesDBRandomChunkBalancedRandomQueryDataset, MoisesDBRandomChunkRandomQueryDataset, \
    MoisesDBDeterministicChunkDeterministicQueryDataset, \
    MoisesDBFullTrackDataset, MoisesDBVDBODeterministicChunkDataset, \
    MoisesDBVDBOFullTrackDataset, MoisesDBVDBORandomChunkDataset, \
    MoisesDBFullTrackTestQueryDataset
    
def MoisesDataModule(
    data_root: str,
    batch_size: int,
    num_workers: int = 8,
    train_kwargs: Optional[Mapping] = None,
    val_kwargs: Optional[Mapping] = None,
    test_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
) -> pl.LightningDataModule:
    if train_kwargs is None:
        train_kwargs = {}

    if val_kwargs is None:
        val_kwargs = {}

    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}

    train_dataset = MoisesDBRandomChunkRandomQueryDataset(
        data_root=data_root, split="train", **train_kwargs
    )

    val_dataset = MoisesDBDeterministicChunkDeterministicQueryDataset(
        data_root=data_root, split="val", **val_kwargs
    )

    test_dataset = MoisesDBDeterministicChunkDeterministicQueryDataset(
        data_root=data_root, split="test", **test_kwargs
    )

    datamodule = from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs
    )

    datamodule.predict_dataloader = (  # type: ignore[method-assign]
        datamodule.test_dataloader
    )

    return datamodule

def MoisesBalancedTrainDataModule(
    data_root: str,
    batch_size: int,
    num_workers: int = 8,
    train_kwargs: Optional[Mapping] = None,
    val_kwargs: Optional[Mapping] = None,
    test_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
) -> pl.LightningDataModule:
    if train_kwargs is None:
        train_kwargs = {}

    if val_kwargs is None:
        val_kwargs = {}

    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}

    train_dataset = MoisesDBRandomChunkBalancedRandomQueryDataset(
        data_root=data_root, split="train", **train_kwargs
    )

    val_dataset = MoisesDBDeterministicChunkDeterministicQueryDataset(
        data_root=data_root, split="val", **val_kwargs
    )

    test_dataset = MoisesDBDeterministicChunkDeterministicQueryDataset(
        data_root=data_root, split="test", **test_kwargs
    )

    datamodule = from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs
    )

    datamodule.predict_dataloader = (  # type: ignore[method-assign]
        datamodule.test_dataloader
    )

    return datamodule
    

def MoisesValidationDataModule(
    data_root: str,
    batch_size: int,
    num_workers: int = 8,
    val_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
    **kwargs
) -> pl.LightningDataModule:
    if val_kwargs is None:
        val_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}
        
    allowed_stems = val_kwargs.get("allowed_stems", None)
    
    assert allowed_stems is not None, "allowed_stems must be provided"
    
    val_datasets = []
    
    for allowed_stem in allowed_stems:
        kwargs = val_kwargs.copy()
        kwargs["allowed_stems"] = [allowed_stem]
        val_dataset = MoisesDBDeterministicChunkDeterministicQueryDataset(
            data_root=data_root, split="val", 
            **kwargs
        )
        
        val_datasets.append(val_dataset)

    datamodule = from_datasets(
        val_dataset=val_datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs
    )

    datamodule.predict_dataloader = (  # type: ignore[method-assign]
        datamodule.val_dataloader
    )

    return datamodule

def MoisesTestDataModule(
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 8,
    test_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
    **kwargs
) -> pl.LightningDataModule:
    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}
        
    allowed_stems = test_kwargs.get("allowed_stems", None)
    
    assert allowed_stems is not None, "allowed_stems must be provided"

    test_dataset = MoisesDBFullTrackTestQueryDataset(
        data_root=data_root, split="test",
        **test_kwargs
    )

    datamodule = from_datasets(
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs
    )

    datamodule.predict_dataloader = (  # type: ignore[method-assign]
        datamodule.test_dataloader
    )

    return datamodule


def MoisesVDBODataModule(
    data_root: str,
    batch_size: int,
    num_workers: int = 8,
    train_kwargs: Optional[Mapping] = None,
    val_kwargs: Optional[Mapping] = None,
    test_kwargs: Optional[Mapping] = None,
    datamodule_kwargs: Optional[Mapping] = None,
):
    
    
    if train_kwargs is None:
        train_kwargs = {}

    if val_kwargs is None:
        val_kwargs = {}

    if test_kwargs is None:
        test_kwargs = {}

    if datamodule_kwargs is None:
        datamodule_kwargs = {}
        
    train_dataset = MoisesDBVDBORandomChunkDataset(
        data_root=data_root, split="train", **train_kwargs
    )
    
    val_dataset = MoisesDBVDBODeterministicChunkDataset(
        data_root=data_root, split="val", **val_kwargs
    )
    
    test_dataset = MoisesDBVDBOFullTrackDataset(
        data_root=data_root, split="test", **test_kwargs
    )
    
    predict_dataset = test_dataset
    
    datamodule = from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        predict_dataset=predict_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **datamodule_kwargs
    )
    
    return datamodule
    
    