import warnings
from typing import Any, Dict, Optional, Tuple, Union
import pytorch_lightning as pl

#from audiocraft.models import encodec
import torch
from torch import nn
from torch.nn import functional as F

from ...types import (
    BatchedInputOutput,
    InputType,
    OperationMode,
    OutputType,
    SimpleishNamespace,
    TensorCollection
)

import torchaudio as ta


class BaseEndToEndModule(pl.LightningModule):

    def __init__(
        self,
    ) -> None:
        super().__init__()


if __name__ == "__main__":
    model = BaseEndToEndModule()
    print(model)
    print(model.__class__.__name__)
    print(model.__module__)
