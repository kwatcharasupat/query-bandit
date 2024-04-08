import math
import torch
from core.models.e2e.conditioners.base import Conditioning


from torch import nn
from torch.nn.modules import activation as activation_


class FiLM(Conditioning):
    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        additive: bool = True,
        multiplicative: bool = False,
        depth: int = 1,
        activation: str = "ELU",
        channels_per_group: int = 16,
    ):
        super().__init__(
            channels=channels,
            channels_per_group=channels_per_group,
            cond_embedding_dim=cond_embedding_dim,
        )

        self.additive = additive
        self.multiplicative = multiplicative
        self.depth = depth
        self.activation = activation

        Activation = activation_.__dict__[activation]

        if self.multiplicative:

            if depth == 1:
                self.gamma = nn.Linear(self.cond_embedding_dim, self.channels)
            else:
                layers = [nn.Linear(self.cond_embedding_dim, self.channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(self.channels, self.channels)]
                self.gamma = nn.Sequential(*layers)
        else:
            self.gamma = None

        if self.additive:
            if depth == 1:
                self.beta = nn.Linear(self.cond_embedding_dim, self.channels)
            else:
                layers = [nn.Linear(self.cond_embedding_dim, self.channels)]
                for _ in range(depth - 1):
                    layers += [Activation(), nn.Linear(self.channels, self.channels)]
                self.beta = nn.Sequential(*layers)
        else:
            self.beta = None

    def forward(self, x, w):

        x = self.gn(x)

        if self.multiplicative:
            gamma = self.gamma(w)

            if len(x.shape) == 4:
                gamma = gamma[:, :, None, None]
            elif len(x.shape) == 3:
                gamma = gamma[:, :, None]
            elif len(x.shape) == 2:
                pass
            else:
                raise ValueError(f"Invalid shape for input tensor: {x.shape}")

            x = gamma * x

        if self.additive:
            beta = self.beta(w)
            if len(x.shape) == 4:
                beta = beta[:, :, None, None]
            elif len(x.shape) == 3:
                beta = beta[:, :, None]
            elif len(x.shape) == 2:
                pass
            else:
                raise ValueError(f"Invalid shape for input tensor: {x.shape}")

            x = x + beta

        return x
        
class CosineSimiliarity(Conditioning):
    def __init__(self, cond_embedding_dim: int, channels: int, channels_per_group: int = 16):
        super().__init__(cond_embedding_dim, channels, channels_per_group)
        
        self.csim = nn.CosineSimilarity(dim=1)
        self.proj = nn.Linear(self.cond_embedding_dim, self.channels * self.channels)
        
    def forward(self, x, w):
        
        
        x = self.gn(x)

        gamma = self.gamma(w)

        if len(x.shape) == 4:
            gamma = gamma[:, :, None, None]
        elif len(x.shape) == 3:
            gamma = gamma[:, :, None]
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError(f"Invalid shape for input tensor: {x.shape}")
        
        c = self.csim(gamma, x)
        
        x = c[:, None, ...] * x

        
        


class GeneralizedBilinear(nn.Bilinear):
    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in1_features, in2_features, out_features, bias, device, dtype)
        
    def forward(self, x1, x2):
        
        out = torch.einsum(
            "bc...,acd,bd->ba...", x1, self.weight, x2
        )
        
        if self.bias is not None:
            ndim = out.ndim
            bias = torch.reshape(self.bias, (1, -1) + (1,) * (ndim - 2))
            
            out = out + bias
            
        return out
           

class BilinearFiLM(Conditioning):
    def __init__(
        self,
        cond_embedding_dim: int,
        channels: int,
        additive: bool = True,
        multiplicative: bool = False,
        depth: int = 2,
        activation: str = "ELU",
        channels_per_group: int = 16,
    ):
        super().__init__(
            channels=channels,
            channels_per_group=channels_per_group,
            cond_embedding_dim=cond_embedding_dim,
        )

        self.additive = additive
        self.multiplicative = multiplicative
        self.depth = depth
        assert depth == 2, "Only depth 2 is supported for BilinearFiLM"
        self.activation = activation

        Activation = activation_.__dict__[activation]

        if self.multiplicative:
            self.gamma_proj = nn.Sequential(
                nn.Linear(self.cond_embedding_dim, self.channels),
                Activation(),
            )
            self.gamma_bilinear = GeneralizedBilinear(self.channels, self.channels, self.channels)
        else:
            self.gamma = None

        if self.additive:
            self.beta_proj = nn.Sequential(
                nn.Linear(self.cond_embedding_dim, self.channels),
                Activation(),
            )
            self.beta_bilinear = GeneralizedBilinear(self.channels, self.channels, self.channels)
        else:
            self.beta = None

    def forward(self, x, w):

        x = self.gn(x)

        if self.multiplicative:
            gamma = self.gamma_proj(w)
            gamma = self.gamma_bilinear(x, gamma)
            x = gamma * x

        if self.additive:
            beta = self.beta_proj(w)
            beta = self.beta_bilinear(x, beta)
            x = x + beta

        return x