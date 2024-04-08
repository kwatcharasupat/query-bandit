from typing import Dict, List, Optional, Union
from torch import nn
import torch
from torch.nn.modules.loss import _Loss

from core.types import BatchedInputOutput
from torch.nn import functional as F

class BaseLossHandler(nn.Module):
    def __init__(
        self, loss: nn.Module, modality: Union[str, List[str]], name: Optional[str] = None
    ) -> None:
        super().__init__()

        self.loss = loss

        if isinstance(modality, str):
            modality = [modality]

        self.modality = modality

        if name is None:
            name = "loss"

        if name == "__auto__":
            name = self.loss.__class__.__name__

        self.name = name

    def _audio_preprocess(self, y_pred, y_true):

        n_sample_true = y_true.shape[-1]
        n_sample_pred = y_pred.shape[-1]

        if n_sample_pred > n_sample_true:
            y_pred = y_pred[..., :n_sample_true]
        elif n_sample_pred < n_sample_true:
            y_true = y_true[..., :n_sample_pred]

        return y_pred, y_true

    def forward(self, batch: BatchedInputOutput):
        y_true = batch.sources
        y_pred = batch.estimates

        loss_contribs = {}

        stem_contribs = {
            stem: 0.0 for stem in y_pred.keys()
        }

        for stem in y_pred.keys():
            for modality in self.modality:

                if modality not in y_pred[stem].keys():
                    continue

                if y_pred[stem][modality].shape[-1] == 0:
                    continue

                y_true_ = y_true[stem][modality]
                y_pred_ = y_pred[stem][modality]

                if modality == "audio":
                    y_pred_, y_true_ = self._audio_preprocess(y_pred_, y_true_)
                elif modality == "spectrogram":
                    y_pred_ = torch.view_as_real(y_pred_)
                    y_true_ = torch.view_as_real(y_true_)

                loss_contribs[f"{self.name}/{stem}/{modality}"] = self.loss(
                    y_pred_, y_true_
                )

                stem_contribs[stem] += loss_contribs[f"{self.name}/{stem}/{modality}"]

        total_loss = sum(stem_contribs.values())
        loss_contribs[self.name] = total_loss

        with torch.no_grad():
            for stem in stem_contribs.keys():
                loss_contribs[f"{self.name}/{stem}"] = stem_contribs[stem]

        return loss_contribs


class AdversarialLossHandler(BaseLossHandler):
    def __init__(self, loss: nn.Module, modality: str, name: Optional[str] = "adv_loss"):

        super().__init__(loss, modality, name)

    def discriminator_forward(self, batch: BatchedInputOutput):

        y_true = batch.sources
        y_pred = batch.estimates

        # g_loss_contribs = {}
        d_loss_contribs = {}

        for stem in y_pred.keys():

            if self.modality not in y_pred[stem].keys():
                continue

            if y_pred[stem][self.modality].shape[-1] == 0:
                continue

            y_true_ = y_true[stem][self.modality]
            y_pred_ = y_pred[stem][self.modality]

            if self.modality == "audio":
                y_pred_, y_true_ = self._audio_preprocess(y_pred_, y_true_)

            # g_loss_contribs[f"{self.name}:g/{stem}"] = self.loss.generator_loss(
            #     y_pred_, y_true_
            # )

            d_loss_contribs[f"{self.name}:d/{stem}"] = self.loss.discriminator_loss(
                y_pred_, y_true_
            )

        # g_total_loss = sum(g_loss_contribs.values())
        d_total_loss = sum(d_loss_contribs.values())

        # g_loss_contribs["loss"] = g_total_loss
        d_loss_contribs["disc_loss"] = d_total_loss

        return d_loss_contribs

    def generator_forward(self, batch: BatchedInputOutput):

        y_true = batch.sources
        y_pred = batch.estimates

        g_loss_contribs = {}
        # d_loss_contribs = {}

        for stem in y_pred.keys():

            if self.modality not in y_pred[stem].keys():
                continue

            if y_pred[stem][self.modality].shape[-1] == 0:
                continue

            y_true_ = y_true[stem][self.modality]
            y_pred_ = y_pred[stem][self.modality]

            if self.modality == "audio":
                y_pred_, y_true_ = self._audio_preprocess(y_pred_, y_true_)

            g_loss_contribs[f"{self.name}:g/{stem}"] = self.loss.generator_loss(
                y_pred_, y_true_
            )

            # d_loss_contribs[f"{self.name}:g/{stem}"] = self.loss.discriminator_loss(
            #     y_pred_, y_true_
            # )

        g_total_loss = sum(g_loss_contribs.values())
        # d_total_loss = sum(d_loss_contribs.values())

        g_loss_contribs["gen_loss"] = g_total_loss
        # d_loss_contribs["loss"] = d_total_loss

        return g_loss_contribs

    def forward(self, batch: BatchedInputOutput):
        return {
            "generator": self.generator_forward(batch),
            "discriminator": self.discriminator_forward(batch)
        }
