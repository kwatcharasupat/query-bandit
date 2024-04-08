import torch
import torchaudio as ta
from hear21passt.base import get_basic_model
from torch import nn

class Passt(nn.Module):

    PASST_EMB_DIM: int = 768
    PASST_FS: int = 32000

    def __init__(
        self,
        original_fs: int=44100,
        passt_fs: int=PASST_FS,
    ):
        super().__init__()

        self.passt = get_basic_model(mode="embed_only", arch="openmic").eval()
        self.resample = ta.transforms.Resample(
            orig_freq=original_fs, new_freq=passt_fs
        ).eval()

        for p in self.passt.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the PasstWrapper model.

        Args:
            qspec (torch.Tensor): Query spectrogram.
            qaudio (torch.Tensor): Query audio.

        Returns:
            torch.Tensor: Embedding output.
        """
        with torch.no_grad():
            x = torch.mean(x, dim=1)
            x = self.resample(x)

            specs = self.passt.mel(x)[..., :998]
            specs = specs[:, None, ...]
            _, z = self.passt.net(specs)

        return z


class PasstWrapper(nn.Module):

    PASST_EMB_DIM: int = 768
    PASST_FS: int = 32000

    def __init__(
        self,
        cond_emb_dim: int = 384,
        original_cond_emb_dim=PASST_EMB_DIM,
        original_fs: int=44100,
        passt_fs: int=PASST_FS,
    ):
        super().__init__()
        self.cond_emb_dim = cond_emb_dim

        self.passt = get_basic_model(mode="embed_only", arch="openmic").eval()
        self.proj = nn.Linear(original_cond_emb_dim, cond_emb_dim) if cond_emb_dim is not None else nn.Identity()
        self.resample = ta.transforms.Resample(
            orig_freq=original_fs, new_freq=passt_fs
        ).eval()

        for p in self.passt.parameters():
            p.requires_grad = False

    def forward(self, qspec, qaudio):
        """
        Forward pass of the PasstWrapper model.

        Args:
            qspec (torch.Tensor): Query spectrogram.
            qaudio (torch.Tensor): Query audio.

        Returns:
            torch.Tensor: Embedding output.
        """
        with torch.no_grad():
            x = torch.mean(qaudio, dim=1)
            x = self.resample(x)

            specs = self.passt.mel(x)[..., :998]
            specs = specs[:, None, ...]
            _, z = self.passt.net(specs)

        z = self.proj(z)

        return z