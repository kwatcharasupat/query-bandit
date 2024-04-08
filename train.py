import json
import os.path
from pprint import pprint
import random
import string
from types import SimpleNamespace
import pandas as pd

import torch
from tqdm import tqdm

from core.data.moisesdb.datamodule import (
    MoisesTestDataModule,
    MoisesValidationDataModule,
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
)
from core.losses.base import AdversarialLossHandler, BaseLossHandler
from core.losses.l1snr import (
    L1SNRDecibelMatchLoss,
    L1SNRLoss,
    WeightedL1Loss,
    L1SNRLossIgnoreSilence,
)
from core.metrics.base import BaseMetricHandler, MultiModeMetricHandler
from core.metrics.snr import (
    SafeScaleInvariantSignalNoiseRatio,
    SafeSignalNoiseRatio,
    PredictedDecibels,
    TargetDecibels,
)
from core.models.ebase import EndToEndLightningSystem
from core.models.e2e.resunet.resunet import (
    ResUNetPasstConditionedSeparator,
    ResUNetResQueryConditionedSeparator,
    # StupidNet
)

from core.models.e2e.bandit.bandit import Bandit, PasstFiLMConditionedBandit

from omegaconf import OmegaConf
from core.types import LossHandler, OptimizationBundle

from torch import nn, optim
from torch.optim import lr_scheduler

import torchmetrics as tm

import pytorch_lightning as pl
import pytorch_lightning.callbacks
import pytorch_lightning.loggers
from pytorch_lightning.profilers import AdvancedProfiler

import torch.backends.cudnn

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def _allowed_classes_to_dict(allowed_classes):
    return {cls.__name__: cls for cls in allowed_classes}


ALLOWED_MODELS = [
    ResUNetPasstConditionedSeparator,
    ResUNetResQueryConditionedSeparator,
    # StupidNet
    Bandit,
    PasstFiLMConditionedBandit,
]

ALLOWED_MODELS_DICT = _allowed_classes_to_dict(ALLOWED_MODELS)

ALLOWED_DATAMODULES = [
    MoisesDataModule,
    MoisesBalancedTrainDataModule,
    MoisesVDBODataModule,
    MoisesValidationDataModule,
    MoisesTestDataModule,
]

ALLOWED_DATAMODULE_DICT = _allowed_classes_to_dict(ALLOWED_DATAMODULES)

ALLOWED_LOSSES = [
    L1SNRLoss,
    WeightedL1Loss,
    L1SNRDecibelMatchLoss,
    L1SNRLossIgnoreSilence,
]

ALLOWED_LOSS_DICT = _allowed_classes_to_dict(ALLOWED_LOSSES)


def _build_model(config: OmegaConf) -> nn.Module:

    model_config = config.model

    model_name = model_config.cls
    kwargs = model_config.get("kwargs", {})

    if model_name in ALLOWED_MODELS_DICT:
        model = ALLOWED_MODELS_DICT[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def _build_inner_loss(config: OmegaConf) -> nn.Module:
    loss_config = config.loss

    loss_name = loss_config.cls
    kwargs = loss_config.get("kwargs", {})

    if loss_name in ALLOWED_LOSS_DICT:
        loss = ALLOWED_LOSS_DICT[loss_name](**kwargs)
    elif loss_name in nn.modules.loss.__dict__:
        loss = nn.modules.loss.__dict__[loss_name](**kwargs)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

    return loss


def _build_loss(config: OmegaConf) -> BaseLossHandler:
    loss_handler = BaseLossHandler(
        loss=_build_inner_loss(config),
        modality=config.loss.modality,
        name=config.loss.get("name", None),
    )

    return loss_handler


def _dummy_metrics(config: OmegaConf) -> MultiModeMetricHandler:
    metrics = MultiModeMetricHandler(
        train_metrics={
            stem: BaseMetricHandler(
                stem=stem,
                metric=tm.MetricCollection(
                    SafeSignalNoiseRatio(),
                    SafeScaleInvariantSignalNoiseRatio(),
                    PredictedDecibels(),
                    TargetDecibels(),
                ),
                modality="audio",
                name="snr",
            )
            for stem in config.stems
        },
        val_metrics={
            stem: BaseMetricHandler(
                stem=stem,
                metric=tm.MetricCollection(
                    SafeSignalNoiseRatio(),
                    SafeScaleInvariantSignalNoiseRatio(),
                    PredictedDecibels(),
                    TargetDecibels(),
                ),
                modality="audio",
                name="snr",
            )
            for stem in config.stems
        },
        test_metrics={
            stem: BaseMetricHandler(
                stem=stem,
                metric=tm.MetricCollection(
                    SafeSignalNoiseRatio(),
                    SafeScaleInvariantSignalNoiseRatio(),
                    PredictedDecibels(),
                    TargetDecibels(),
                ),
                modality="audio",
                name="snr",
            )
            for stem in config.stems
        },
    )

    return metrics


def _build_optimization_bundle(config: OmegaConf) -> OptimizationBundle:
    optim_config = config.optim

    print(optim_config)

    optimizer_name = optim_config.optimizer.cls
    kwargs = optim_config.optimizer.get("kwargs", {})

    optimizer = getattr(optim, optimizer_name)

    optim_bundle = SimpleNamespace(
        optimizer=SimpleNamespace(cls=optimizer, kwargs=kwargs), scheduler=None
    )

    scheduler_config = optim_config.get("scheduler", None)

    if scheduler_config is not None:
        scheduler_name = scheduler_config.cls
        scheduler_kwargs = scheduler_config.get("kwargs", {})

        if scheduler_name in lr_scheduler.__dict__:
            scheduler = lr_scheduler.__dict__[scheduler_name]
        else:
            raise ValueError(f"Unknown scheduler name: {scheduler_name}")

        optim_bundle.scheduler = SimpleNamespace(cls=scheduler, kwargs=scheduler_kwargs)

    return optim_bundle


def _dummy_augmentation():
    return nn.Identity()


def _load_config(config_path: str) -> OmegaConf:
    config = OmegaConf.load(config_path)

    config_dict = {}

    for k, v in config.items():
        if isinstance(v, str) and v.endswith(".yml"):
            config_dict[k] = OmegaConf.load(v)
        else:
            config_dict[k] = v

    config = OmegaConf.merge(config_dict)

    return config


def _build_datamodule(config: OmegaConf) -> pl.LightningDataModule:

    DataModule = ALLOWED_DATAMODULE_DICT[config.data.cls]

    datamodule = DataModule(
        data_root=config.data.data_root,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_kwargs=config.data.get("train_kwargs", None),
        val_kwargs=config.data.get("val_kwargs", None),
        test_kwargs=config.data.get("test_kwargs", None),
        datamodule_kwargs=config.data.get("datamodule_kwargs", None),
    )

    return datamodule


def train(
    config_path: str,
    profile: bool = False,
    ckpt_path: str = None,
    validate_only: bool = False,
    inference_only: bool = False,
    output_dir: str = None,
    test_datamodule: bool = False,
    precision=32,
):
    config = _load_config(config_path)

    pl.seed_everything(config.seed, workers=True)

    if inference_only:
        config["data"]["batch_size"] = 1

    datamodule = _build_datamodule(config)

    if test_datamodule:
        for batch in tqdm(datamodule.train_dataloader()):
            pass

        for batch in tqdm(datamodule.val_dataloader()):
            pass

        for batch in tqdm(datamodule.test_dataloader()):
            pass

        return

    model = _build_model(config)
    loss_handler = _build_loss(config)

    system = EndToEndLightningSystem(
        model=model,
        loss_handler=loss_handler,
        metrics=_dummy_metrics(config),
        augmentation_handler=_dummy_augmentation(),
        inference_handler=config.get("inference", None),
        optimization_bundle=_build_optimization_bundle(config),
        fast_run=config.fast_run,
        batch_size=config.data.batch_size,
        effective_batch_size=config.data.get("effective_batch_size", None),
        commitment_weight=config.get("commitment_weight", 1.0),
    )

    rand_str = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
    )

    logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=os.path.join(
            config.trainer.logger.save_dir, os.environ.get("SLURM_JOB_ID", rand_str)
        ),
    )

    callbacks = [
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor=config.trainer.callbacks.checkpoint.monitor,
            mode=config.trainer.callbacks.checkpoint.mode,
            save_top_k=config.trainer.callbacks.checkpoint.save_top_k,
            save_last=config.trainer.callbacks.checkpoint.save_last,
        ),
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor=None,
        ),  # also save the last 3 epochs
        pytorch_lightning.callbacks.RichModelSummary(max_depth=3),
    ]

    if profile:
        profiler = AdvancedProfiler(filename="profile.txt", dirpath=".")

    if config.trainer.accumulate_grad_batches is None:
        config.trainer.accumulate_grad_batches = 1
        if config.data.effective_batch_size is not None:
            config.trainer.accumulate_grad_batches = int(
                config.data.effective_batch_size / config.data.batch_size
            )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=1 if profile else config.trainer.max_epochs,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler if profile else None,
        limit_train_batches=int(8) if profile else float(1.0),
        limit_val_batches=int(8) if profile else float(1.0),
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        precision=precision,
        gradient_clip_val=config.trainer.get("gradient_clip_val", None),
        gradient_clip_algorithm=config.trainer.get("gradient_clip_algorithm", "norm"),
    )

    if validate_only:
        trainer.validate(system, datamodule, ckpt_path=ckpt_path)
    elif inference_only:
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(os.path.dirname(ckpt_path)), "inference"
            )
            system.set_output_path(output_dir)
        trainer.predict(system, datamodule, ckpt_path=ckpt_path)
    else:
        trainer.logger.log_hyperparams(OmegaConf.to_object(config))
        trainer.logger.save()
        trainer.fit(system, datamodule, ckpt_path=ckpt_path)


def query_validate(config_path: str, ckpt_path: str):
    config = _load_config(config_path)

    datamodule = _build_datamodule(config)

    model = _build_model(config)
    loss_handler = _build_loss(config)

    system = EndToEndLightningSystem(
        model=model,
        loss_handler=loss_handler,
        metrics=_dummy_metrics(config),
        augmentation_handler=_dummy_augmentation(),
        inference_handler=None,
        optimization_bundle=_build_optimization_bundle(config),
        fast_run=config.fast_run,
        batch_size=config.data.batch_size,
        effective_batch_size=config.data.get("effective_batch_size", None),
        commitment_weight=config.get("commitment_weight", 1.0),
    )

    logger = pytorch_lightning.loggers.CSVLogger(
        save_dir=os.path.join(config.trainer.logger.save_dir, "validate"),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    allowed_stems = config.data.val_kwargs.get("allowed_stems", None)

    data = []

    os.makedirs(trainer.logger.log_dir, exist_ok=True)

    with open(trainer.logger.log_dir + "/config.txt", "w") as f:
        f.write(ckpt_path)

    dl = datamodule.val_dataloader()

    for stem, val_dl in zip(allowed_stems, dl):
        metrics = trainer.validate(system, val_dl, ckpt_path=ckpt_path)[0]
        print(stem)
        pprint(metrics)

        for metric, value in metrics.items():
            data.append({"metric": metric, "value": value, "stem": stem})

    df = pd.DataFrame(data)

    df.to_csv(
        os.path.join(trainer.logger.log_dir, "validation_metrics.csv"), index=False
    )


def query_test(config_path: str, ckpt_path: str):
    config = _load_config(config_path)

    pprint(config)
    pprint(config.data.inference_kwargs)

    datamodule = _build_datamodule(config)

    model = _build_model(config)
    loss_handler = _build_loss(config)

    system = EndToEndLightningSystem(
        model=model,
        loss_handler=loss_handler,
        metrics=_dummy_metrics(config),
        augmentation_handler=_dummy_augmentation(),
        inference_handler=config.data.inference_kwargs,
        optimization_bundle=_build_optimization_bundle(config),
        fast_run=config.fast_run,
        batch_size=config.data.batch_size,
        effective_batch_size=config.data.get("effective_batch_size", None),
        commitment_weight=config.get("commitment_weight", 1.0),
    )

    rand_str = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
    )

    use_own_query = config.data.test_kwargs.get("use_own_query", False)

    prefix = "test-o" if use_own_query else "test"

    logger = pytorch_lightning.loggers.CSVLogger(
        save_dir=os.path.join(
            config.trainer.logger.save_dir,
            prefix,
            os.environ.get("SLURM_JOB_ID", rand_str),
        ),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)

    with open(trainer.logger.log_dir + "/config.txt", "w") as f:
        f.write(ckpt_path)

    trainer.logger.log_hyperparams(OmegaConf.to_object(config))
    trainer.logger.save()

    dl = datamodule.test_dataloader()

    trainer.test(system, dl, ckpt_path=ckpt_path)

def query_inference(config_path: str, ckpt_path: str):
    config = _load_config(config_path)

    pprint(config)
    pprint(config.data.inference_kwargs)

    datamodule = _build_datamodule(config)

    model = _build_model(config)
    loss_handler = _build_loss(config)

    system = EndToEndLightningSystem(
        model=model,
        loss_handler=loss_handler,
        metrics=_dummy_metrics(config),
        augmentation_handler=_dummy_augmentation(),
        inference_handler=config.data.inference_kwargs,
        optimization_bundle=_build_optimization_bundle(config),
        fast_run=config.fast_run,
        batch_size=config.data.batch_size,
        effective_batch_size=config.data.get("effective_batch_size", None),
        commitment_weight=config.get("commitment_weight", 1.0),
    )

    rand_str = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
    )

    use_own_query = config.data.test_kwargs.get("use_own_query", False)

    prefix = "inference-o" if use_own_query else "inference-d"

    logger = pytorch_lightning.loggers.CSVLogger(
        save_dir=os.path.join(
            config.trainer.logger.save_dir,
            prefix,
            os.environ.get("SLURM_JOB_ID", rand_str),
        ),
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
    )

    os.makedirs(trainer.logger.log_dir, exist_ok=True)

    with open(trainer.logger.log_dir + "/config.txt", "w") as f:
        f.write(ckpt_path)

    trainer.logger.log_hyperparams(OmegaConf.to_object(config))
    trainer.logger.save()

    dl = datamodule.test_dataloader()

    trainer.predict(system, dl, ckpt_path=ckpt_path)


def clean_validation_metrics(path):
    df = pd.read_csv(path).T

    data = []

    stems = [
        "drums",
        "lead_male_singer",
        "lead_female_singer",
        # "human_choir",
        "background_vocals",
        # "other_vocals",
        "bass_guitar",
        "bass_synthesizer",
        # "contrabass_double_bass",
        # "tuba",
        # "bassoon",
        "fx",
        "clean_electric_guitar",
        "distorted_electric_guitar",
        # "lap_steel_guitar_or_slide_guitar",
        "acoustic_guitar",
        "other_plucked",
        "pitched_percussion",
        "grand_piano",
        "electric_piano",
        "organ_electric_organ",
        "synth_pad",
        "synth_lead",
        # "violin",
        # "viola",
        # "cello",
        # "violin_section",
        # "viola_section",
        # "cello_section",
        "string_section",
        "other_strings",
        "brass",
        # "flutes",
        "reeds",
        "other_wind",
    ]

    for metric, value in df.iterrows():

        mm = metric.split("/")
        idx = mm[-1]
        m = "/".join(mm[:-1])

        print(metric, idx)

        try:
            idx = int(idx.split("_")[-1])
        except ValueError as e:
            assert "invalid literal for int() with base 10" in str(e)
            continue

        data.append({m: value, "stem": stems[idx]})

    df = pd.DataFrame(data)

    new_path = path.replace(".csv", "_clean.csv")

    df.to_csv(new_path, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire()
