from typing import Dict, Optional
from torch import nn
import torchmetrics as tm

from core.types import BatchedInputOutput, OperationMode


class BaseMetricHandler(nn.Module):
    def __init__(
        self, stem: str, metric: tm.Metric, modality: str, name: Optional[str] = None
    ) -> None:
        super().__init__()

        self.metric = metric
        self.modality = modality
        self.stem = stem

        if name is None or name == "__auto__":
            name = self.metric.__class__.__name__

        self.name = name

    def update(self, batch: BatchedInputOutput):
        y_true = batch.sources[self.stem]
        y_pred = batch.estimates[self.stem]

        self.metric.update(y_pred[self.modality].cuda(), y_true[self.modality].cuda())

    def compute(self) -> Dict[str, float]:

        metric = self.metric.compute()

        if isinstance(metric, dict):
            return {f"{self.name}/{k}": v for k, v in metric.items()}

        return {self.name: self.metric.compute()}

    def reset(self):
        self.metric.reset()


class MultiModeMetricHandler(nn.Module):
    def __init__(
        self,
        train_metrics: Dict[str, BaseMetricHandler],
        val_metrics: Dict[str, BaseMetricHandler],
        test_metrics: Dict[str, BaseMetricHandler],
    ):
        super().__init__()

        self.train_metrics = nn.ModuleDict(train_metrics)
        self.val_metrics = nn.ModuleDict(val_metrics)
        self.test_metrics = nn.ModuleDict(test_metrics)

    def get_mode(self, mode: OperationMode) -> BaseMetricHandler:
        if mode == OperationMode.TRAIN:
            return self.train_metrics
        elif mode == OperationMode.VAL:
            return self.val_metrics
        elif mode == OperationMode.TEST:
            return self.test_metrics
        else:
            raise ValueError(f"Unknown mode: {mode}")
