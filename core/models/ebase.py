import math
import os.path
from collections import defaultdict
from itertools import chain, combinations
from pprint import pprint
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple, Type, TypedDict

import pytorch_lightning as pl
import torch
import torchaudio as ta
import torchmetrics as tm
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from torch.nn import functional as F

from core.types import BatchedInputOutput, OperationMode, RawInputType, SimpleishNamespace
from core.types import (
    InputType,
    OutputType,
    LossOutputType,
    MetricOutputType,
    ModelType,
    OptimizerType,
    SchedulerType,
    MetricType,
    LossType,
    OptimizationBundle,
    LossHandler,
    MetricHandler,
    AugmentationHandler,
    InferenceHandler,
)


class EndToEndLightningSystem(pl.LightningModule):
    def __init__(
        self,
        model: ModelType,
        loss_handler: LossHandler,
        metrics: MetricHandler,
        augmentation_handler: AugmentationHandler,
        inference_handler: InferenceHandler,
        optimization_bundle: OptimizationBundle,
        fast_run: bool = False,
        commitment_weight: float = 1.0,
        batch_size: Optional[int] = None,
        effective_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = model

        self.loss = loss_handler

        self.metrics = metrics
        self.optimization = optimization_bundle
        self.augmentation = augmentation_handler
        self.inference = inference_handler

        self.fast_run = fast_run

        self.model.fast_run = fast_run

        self.commitment_weight = commitment_weight

        self.batch_size = batch_size
        self.effective_batch_size = effective_batch_size if effective_batch_size is not None else batch_size
        self.accum_ratio = self.effective_batch_size // self.batch_size if self.effective_batch_size is not None else 1

        self.output_dir = None
        self.split_size = None

    def configure_optimizers(self) -> Any:
        optimizer = self.optimization.optimizer.cls(
            self.model.parameters(),
            **self.optimization.optimizer.kwargs
        )

        ret = {
            "optimizer": optimizer,
        }

        if self.optimization.scheduler is not None:
            scheduler = self.optimization.scheduler.cls(
                optimizer,
                **self.optimization.scheduler.kwargs
            )
            ret["lr_scheduler"] = scheduler

        return ret

    def compute_loss(
        self,
        batch: BatchedInputOutput,
        mode=OperationMode.TRAIN
    ) -> LossOutputType:
        loss_dict = self.loss(batch)
        return loss_dict

    # TODO: move to a metric handler
    def update_metrics(
        self,
        batch: BatchedInputOutput,
        mode: OperationMode = OperationMode.TRAIN,
    ) -> None:
        metrics: MetricType = self.metrics.get_mode(mode)

        for stem, metric in metrics.items():
            if stem not in batch.estimates.keys():
                continue
            metric.update(batch)

    # TODO: move to a metric handler
    def compute_metrics(self, mode: OperationMode) -> MetricOutputType:
        metrics: MetricType = self.metrics.get_mode(mode)

        metric_dict = {}

        for stem, metric in metrics.items():
            md = metric.compute()
            metric_dict.update({f"{stem}/{k}": v for k, v in md.items()})

        self.log_dict(metric_dict, prog_bar=True, logger=False)

        return metric_dict

    # TODO: move to a metric handler
    def reset_metrics(self, mode: OperationMode) -> None:
        metrics: MetricType = self.metrics.get_mode(mode)

        for _, metric in metrics.items():
            metric.reset()

    def forward(self, batch: RawInputType) -> Tuple[InputType, OutputType]:
        batch = self.model(batch)
        return batch

    def common_step(
        self, batch: RawInputType, mode: OperationMode, batch_idx: int = -1
    ) -> Tuple[OutputType, LossOutputType]:
        batch = BatchedInputOutput.from_dict(batch)
        batch = self.forward(batch)

        loss_dict = self.compute_loss(batch, mode=mode)

        if not self.fast_run:
            with torch.no_grad():
                self.update_metrics(batch, mode=mode)

        return loss_dict

    def training_step(self, batch: RawInputType, batch_idx: int) -> LossOutputType:
        # augmented_batch = self.augmentation(batch, mode=OperationMode.TRAIN)

        self.model.train()

        loss_dict = self.common_step(batch, mode=OperationMode.TRAIN, batch_idx=batch_idx)

        self.log_dict_with_prefix(loss_dict, prefix=OperationMode.TRAIN, prog_bar=True)

        return loss_dict

    def on_train_batch_end(
        self, outputs: OutputType, batch: RawInputType, batch_idx: int
    ) -> None:

        if self.fast_run:
            return

        if (batch_idx + 1) % self.accum_ratio == 0:
            metric_dict = self.compute_metrics(mode=OperationMode.TRAIN)
            self.log_dict_with_prefix(metric_dict, prefix=OperationMode.TRAIN)
            self.reset_metrics(mode=OperationMode.TRAIN)

    @torch.inference_mode()
    def validation_step(
        self, batch: RawInputType, batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:

        self.model.eval()

        with torch.inference_mode():
            loss_dict = self.common_step(batch, mode=OperationMode.VAL)

        self.log_dict_with_prefix(loss_dict, prefix=OperationMode.VAL)

        return loss_dict

    def on_validation_epoch_start(self) -> None:
        self.reset_metrics(mode=OperationMode.VAL)

    def on_validation_epoch_end(self) -> None:
        if self.fast_run:
            return

        metric_dict = self.compute_metrics(mode=OperationMode.VAL)
        self.log_dict_with_prefix(
            metric_dict, OperationMode.VAL, prog_bar=True, add_dataloader_idx=False
        )
        self.reset_metrics(mode=OperationMode.VAL)

    
    def save_to_audio(self, batch: BatchedInputOutput, batch_idx: int) -> None:
        
        batch_size = batch["mixture"]["audio"].shape[0]
        
        assert batch_size == 1, "Batch size must be 1 for inference"
        
        metadata = batch.metadata
        
        song_id = metadata["mix"][0]
        stem = metadata["stem"][0]
        
        log_dir = os.path.join(self.logger.log_dir, "audio")
        
        os.makedirs(os.path.join(log_dir, song_id), exist_ok=True)
        
        audio = batch.estimates[stem]["audio"]
        
        audio = audio.squeeze(0).cpu().numpy()
        
        audio_path = os.path.join(log_dir, song_id, f"{stem}.wav")
        
        ta.save(audio_path, torch.tensor(audio), self.inference.fs)

    def save_vdbo_to_audio(self, batch: BatchedInputOutput, batch_idx: int) -> None:
        
        batch_size = batch["mixture"]["audio"].shape[0]
        
        assert batch_size == 1, "Batch size must be 1 for inference"
        
        metadata = batch.metadata
        
        song_id = metadata["song_id"][0]
        
        log_dir = os.path.join(self.logger.log_dir, "audio")
        
        os.makedirs(os.path.join(log_dir, song_id), exist_ok=True)
        
        for stem, audio in batch.estimates.items():
            audio = audio["audio"]
            audio = audio.squeeze(0).cpu().numpy()
            
            audio_path = os.path.join(log_dir, song_id, f"{stem}.wav")
            
            ta.save(audio_path, torch.tensor(audio), self.inference.fs)

    @torch.inference_mode()
    def chunked_inference(
        self, batch: RawInputType, batch_idx: int = -1, dataloader_idx: int = 0
    ) -> BatchedInputOutput:
        batch = BatchedInputOutput.from_dict(batch)
        
        audio = batch["mixture"]["audio"]
        
        b, c, n_samples = audio.shape
        
        assert b == 1

        fs = self.inference.fs

        chunk_size = int(self.inference.chunk_size_seconds * fs)
        hop_size = int(self.inference.hop_size_seconds * fs)
        
        batch_size = self.inference.batch_size
        
        overlap = chunk_size - hop_size
        
        scaler = chunk_size / (2 * hop_size)

        n_chunks = int(math.ceil(
            (n_samples + 4 * overlap - chunk_size) / hop_size
        )) + 1
        
        pad = (n_chunks - 1) * hop_size + chunk_size - n_samples

        # print(audio.shape)
        audio = F.pad(
            audio,
            pad=(2 * overlap, 2 * overlap + pad),
            mode="reflect"
        )
        padded_length = audio.shape[-1]
        audio = audio.reshape(c, 1, -1, 1)
        
        chunked_audio = F.unfold(
            audio,
            kernel_size=(chunk_size, 1), 
            stride=(hop_size, 1)
        ) # (c, chunk_size, n_chunk)

        # print(chunked_audio.shape)

        chunked_audio = chunked_audio.permute(2, 0, 1).reshape(-1, c, chunk_size)
        
        n_chunks = chunked_audio.shape[0]
        
        n_batch = math.ceil(n_chunks / batch_size)

        outputs = []
        
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_chunks)
            
            chunked_batch = SimpleishNamespace(
                mixture={
                    "audio": chunked_audio[start:end]
                },
                query=batch["query"],
                estimates=batch["estimates"]
            )
            
            output = self.forward(chunked_batch)
            outputs.append(output.estimates["target"]["audio"])

        output = torch.cat(outputs, dim=0) # (n_chunks, c, chunk_size)
        window = torch.hann_window(chunk_size, device=self.device).reshape(1, 1, chunk_size)
        output = output * window / scaler

        output = torch.permute(output, (1, 2, 0))

        output = F.fold(
            output,
            output_size=(padded_length, 1),
            kernel_size=(chunk_size, 1),
            stride=(hop_size, 1)
        ) # (c, 1, t, 1)

        output = output[None, :, 0, 2*overlap: n_samples + 2*overlap, 0]

        stem = batch.metadata["stem"][0]

        batch["estimates"][stem] = {
            "audio": output
        }

        return batch

    def chunked_vdbo_inference(
        self, batch: RawInputType, batch_idx: int = -1, dataloader_idx: int = 0
    ) -> BatchedInputOutput:
        batch = BatchedInputOutput.from_dict(batch)
        
        audio = batch["mixture"]["audio"]
        
        b, c, n_samples = audio.shape
        
        assert b == 1

        fs = self.inference.fs

        chunk_size = int(self.inference.chunk_size_seconds * fs)
        hop_size = int(self.inference.hop_size_seconds * fs)
        
        batch_size = self.inference.batch_size
        
        overlap = chunk_size - hop_size
        
        scaler = chunk_size / (2 * hop_size)

        n_chunks = int(math.ceil(
            (n_samples + 4 * overlap - chunk_size) / hop_size
        )) + 1
        
        pad = (n_chunks - 1) * hop_size + chunk_size - n_samples

        # print(audio.shape)
        audio = F.pad(
            audio,
            pad=(2 * overlap, 2 * overlap + pad),
            mode="reflect"
        )
        padded_length = audio.shape[-1]
        audio = audio.reshape(c, 1, -1, 1)
        
        chunked_audio = F.unfold(
            audio,
            kernel_size=(chunk_size, 1), 
            stride=(hop_size, 1)
        ) # (c, chunk_size, n_chunk)

        # print(chunked_audio.shape)

        chunked_audio = chunked_audio.permute(2, 0, 1).reshape(-1, c, chunk_size)
        
        n_chunks = chunked_audio.shape[0]
        
        n_batch = math.ceil(n_chunks / batch_size)

        outputs = defaultdict(list)
        
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_chunks)
            
            chunked_batch = SimpleishNamespace(
                mixture={
                    "audio": chunked_audio[start:end]
                },
                estimates=batch["estimates"]
            )
            
            output = self.forward(chunked_batch)
            
            for stem, estimate in output.estimates.items():
                outputs[stem].append(estimate["audio"])

        for stem, outputs_ in outputs.items():                

            output = torch.cat(outputs_, dim=0) # (n_chunks, c, chunk_size)
            window = torch.hann_window(chunk_size, device=self.device).reshape(1, 1, chunk_size)
            output = output * window / scaler

            output = torch.permute(output, (1, 2, 0))

            output = F.fold(
                output,
                output_size=(padded_length, 1),
                kernel_size=(chunk_size, 1),
                stride=(hop_size, 1)
            ) # (c, 1, t, 1)

            output = output[None, :, 0, 2*overlap: n_samples + 2*overlap, 0]

            batch["estimates"][stem] = {
                "audio": output
            }

        return batch


    def on_test_epoch_start(self) -> None:
        self.reset_metrics(mode=OperationMode.TEST)

    def test_step(
        self, batch: RawInputType, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:

        self.model.eval()
        
        if "query" in batch.keys():
            batch = self.chunked_inference(batch, batch_idx, dataloader_idx)
        else:
            batch = self.chunked_vdbo_inference(batch, batch_idx, dataloader_idx)
        
        self.reset_metrics(mode=OperationMode.TEST)
        self.update_metrics(batch, mode=OperationMode.TEST)
        metrics = self.compute_metrics(mode=OperationMode.TEST)
        # metrics["song_id"] = batch.metadata["mix"][0]
        self.log_dict_with_prefix(metrics, OperationMode.TEST, 
                                  on_step=True, on_epoch=False, prog_bar=True)
        self.reset_metrics(mode=OperationMode.TEST)

        # pprint(metrics)

        return batch

    def on_test_epoch_end(self) -> None:
        self.reset_metrics(mode=OperationMode.TEST)

    def set_output_path(self, output_dir: str) -> None:
        self.output_dir = output_dir

    def predict_step(
        self, batch: RawInputType, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:

        self.model.eval()
        
        if "query" in batch.keys():    
            batch = self.chunked_inference(batch, batch_idx, dataloader_idx)
            
            self.save_to_audio(batch, batch_idx)
        else:
            batch = self.chunked_vdbo_inference(batch, batch_idx, dataloader_idx)
            self.save_vdbo_to_audio(batch, batch_idx)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = False
    ) -> Any:
        return super().load_state_dict(state_dict, strict=False)

    def log_dict_with_prefix(
        self,
        dict_: Dict[str, torch.Tensor],
        prefix: str,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:

        
        self.log_dict(
            {f"{prefix}/{k}": v for k, v in dict_.items()},
            batch_size=batch_size,
            logger=True,
            sync_dist=True,
            **kwargs,
            # on_step=True,
            # on_epoch=False,
        )

        self.logger.save()
