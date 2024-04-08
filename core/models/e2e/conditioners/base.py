from torch import nn


class Conditioning(nn.Module):
    def __init__(
        self, cond_embedding_dim: int, channels: int, channels_per_group: int = 16
    ):
        super().__init__()

        self.channels = channels
        self.cond_embedding_dim = cond_embedding_dim
        self.channels_per_group = channels_per_group

        self.gn = nn.GroupNorm(self.channels // self.channels_per_group, self.channels)

    def forward(self, x, w):
        raise NotImplementedError


class PassThroughConditioning(Conditioning):
    def __init__(
        self, cond_embedding_dim: int, channels: int, channels_per_group: int = 16
    ):
        super().__init__(cond_embedding_dim, channels, channels_per_group)

    def forward(self, x, w):
        return self.gn(x)