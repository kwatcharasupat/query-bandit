from typing import Dict, List, Optional, Tuple
from core.models.e2e.bandit.bandsplit import BandSplitModule
from core.models.e2e.bandit.maskestim import OverlappingMaskEstimationModule
from core.models.e2e.bandit.tfmodel import SeqBandModellingModule
from core.models.e2e.bandit.utils import MusicalBandsplitSpecification
from core.models.e2e.querier.passt import Passt, PasstWrapper
from core.types import InputType, OperationMode, SimpleishNamespace
from torch import Tensor, nn
import torch

from core.models.e2e.base import BaseEndToEndModule
from core.models.e2e.conditioners.film import FiLM

import torchaudio as ta


class BaseBandit(BaseEndToEndModule):

    def __init__(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        n_fft: int = 2048,
        win_length: Optional[int] = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        power: Optional[int] = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        fs: int = 44100,
    ):
        super().__init__()

        self.instantitate_spectral(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

        self.instantiate_bandsplit(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
            n_fft=n_fft,
            fs=fs,
        )

        self.instantiate_tf_modelling(
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )

    def instantitate_spectral(
        self,
        n_fft: int = 2048,
        win_length: Optional[int] = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Optional[Dict] = None,
        power: Optional[int] = None,
        normalized: bool = True,
        center: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
    ):

        assert power is None

        window_fn = torch.__dict__[window_fn]

        self.stft = ta.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

        self.istft = ta.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad_mode=pad_mode,
            pad=0,
            window_fn=window_fn,
            wkwargs=wkwargs,
            normalized=normalized,
            center=center,
            onesided=onesided,
        )

    def instantiate_bandsplit(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        emb_dim: int = 128,
        n_fft: int = 2048,
        fs: int = 44100,
    ):

        assert band_type == "musical"

        self.band_specs = MusicalBandsplitSpecification(
            nfft=n_fft, fs=fs, n_bands=n_bands
        )

        self.band_split = BandSplitModule(
            in_channel=in_channel,
            band_specs=self.band_specs.get_band_specs(),
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            emb_dim=emb_dim,
        )

    def instantiate_tf_modelling(
        self,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ):
        self.tf_model = SeqBandModellingModule(
            n_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
        )

    def mask(self, x, m):
        return x * m

    def forward(self, batch: InputType, mode: OperationMode = OperationMode.TRAIN):

        with torch.no_grad():
            x = self.stft(batch.mixture.audio)
            batch.mixture.spectrogram = x

            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    s = batch.sources[stem].audio
                    s = self.stft(s)
                    batch.sources[stem].spectrogram = s

        batch = self.separate(batch)

        return batch

    def encode(self, batch):
        x = batch.mixture.spectrogram
        length = batch.mixture.audio.shape[-1]

        z = self.band_split(x)  # (batch, emb_dim, n_band, n_time)
        q = self.tf_model(z)  # (batch, emb_dim, n_band, n_time)

        return x, q, length

    def separate(self, batch):
        raise NotImplementedError


class Bandit(BaseBandit):
    def __init__(
        self,
        in_channel: int,
        stems: List[str],
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        fs: int = 44100,
    ):
        super().__init__(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
            fs=fs,
        )

        self.instantiate_mask_estim(
            in_channel=in_channel,
            stems=stems,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )

    def instantiate_mask_estim(
        self,
        in_channel: int,
        stems: List[str],
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = nn.ModuleDict(
            {
                stem: OverlappingMaskEstimationModule(
                    band_specs=self.band_specs.get_band_specs(),
                    freq_weights=self.band_specs.get_freq_weights(),
                    n_freq=n_freq,
                    emb_dim=emb_dim,
                    mlp_dim=mlp_dim,
                    in_channel=in_channel,
                    hidden_activation=hidden_activation,
                    hidden_activation_kwargs=hidden_activation_kwargs,
                    complex_mask=complex_mask,
                    use_freq_weights=use_freq_weights,
                )
                for stem in stems
            }
        )

    def separate(self, batch):

        x, q, length = self.encode(batch)

        for stem, mem in self.mask_estim.items():
            m = mem(q)
            s = self.mask(x, m)
            s = torch.reshape(s, x.shape)
            batch.estimates[stem] = SimpleishNamespace(
                audio=self.istft(s, length), spectrogram=s
            )

        return batch


class BaseConditionedBandit(BaseBandit):
    query_encoder: nn.Module

    def __init__(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        fs: int = 44100,
    ):
        super().__init__(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
            fs=fs,
        )

        self.instantiate_mask_estim(
            in_channel=in_channel,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            n_freq=n_fft // 2 + 1,
            use_freq_weights=use_freq_weights,
        )

    def instantiate_mask_estim(
        self,
        in_channel: int,
        emb_dim: int,
        mlp_dim: int,
        hidden_activation: str,
        hidden_activation_kwargs: Optional[Dict] = None,
        complex_mask: bool = True,
        n_freq: Optional[int] = None,
        use_freq_weights: bool = True,
    ):
        if hidden_activation_kwargs is None:
            hidden_activation_kwargs = {}

        assert n_freq is not None

        self.mask_estim = OverlappingMaskEstimationModule(
            band_specs=self.band_specs.get_band_specs(),
            freq_weights=self.band_specs.get_freq_weights(),
            n_freq=n_freq,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            in_channel=in_channel,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
        )

    def separate(self, batch):

        x, q, length = self.encode(batch)

        q = self.adapt_query(q, batch)

        m = self.mask_estim(q)
        s = self.mask(x, m)
        s = torch.reshape(s, x.shape)
        batch.estimates["target"] = SimpleishNamespace(
            audio=self.istft(s, length), spectrogram=s
        )

        return batch

    def adapt_query(self, q, batch):
        raise NotImplementedError


class PasstFiLMConditionedBandit(BaseConditionedBandit):

    def __init__(
        self,
        in_channel: int,
        band_type: str = "musical",
        n_bands: int = 64,
        additive_film: bool = True,
        multiplicative_film: bool = True,
        film_depth: int = 2,
        require_no_overlap: bool = False,
        require_no_gap: bool = True,
        normalize_channel_independently: bool = False,
        treat_channel_as_feature: bool = True,
        n_sqm_modules: int = 12,
        emb_dim: int = 128,
        rnn_dim: int = 256,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
        mlp_dim: int = 512,
        hidden_activation: str = "Tanh",
        hidden_activation_kwargs: Dict | None = None,
        complex_mask: bool = True,
        use_freq_weights: bool = True,
        n_fft: int = 2048,
        win_length: int | None = 2048,
        hop_length: int = 512,
        window_fn: str = "hann_window",
        wkwargs: Dict | None = None,
        power: int | None = None,
        center: bool = True,
        normalized: bool = True,
        pad_mode: str = "constant",
        onesided: bool = True,
        fs: int = 44100,
        pretrain_encoder = None,
        freeze_encoder = False
    ):
        super().__init__(
            in_channel=in_channel,
            band_type=band_type,
            n_bands=n_bands,
            require_no_overlap=require_no_overlap,
            require_no_gap=require_no_gap,
            normalize_channel_independently=normalize_channel_independently,
            treat_channel_as_feature=treat_channel_as_feature,
            n_sqm_modules=n_sqm_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            mlp_dim=mlp_dim,
            hidden_activation=hidden_activation,
            hidden_activation_kwargs=hidden_activation_kwargs,
            complex_mask=complex_mask,
            use_freq_weights=use_freq_weights,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fn,
            wkwargs=wkwargs,
            power=power,
            center=center,
            normalized=normalized,
            pad_mode=pad_mode,
            onesided=onesided,
            fs=fs,
        )

        self.query_encoder = Passt(
            original_fs=fs,
            passt_fs=32000,
        )
        
        self.film = FiLM(
            self.query_encoder.PASST_EMB_DIM,
            emb_dim,
            additive=additive_film,
            multiplicative=multiplicative_film,
            depth=film_depth,
        )
        
        if pretrain_encoder is not None:
            self.load_pretrained_encoder(pretrain_encoder)
            
            for p in self.band_split.parameters():
                p.requires_grad = not freeze_encoder
                
            for p in self.tf_model.parameters():
                p.requires_grad = not freeze_encoder
            
        
        
    def load_pretrained_encoder(self, path):
        
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        
        state_dict_ = {k.replace("model.", "") if k.startswith("model.") else k: v for k, v in state_dict.items()}
        
        state_dict = {}

        for k, v in state_dict_.items():
            if "mask_estim" in k:
                continue
            
            if "tf_seqband" in k:
                k = k.replace("tf_seqband", "tf_model.seqband")
            
            state_dict[k] = v
            
        
        res = self.load_state_dict(state_dict, strict=False)
        
        for k in res.unexpected_keys:
            if "mask_estim" in k:
                continue
            print(f"Unexpected key: {k}")
        
        for k in res.missing_keys:
            print(f"Missing key: {k}")
            for kw in ["band_split", "tf_model"]:
                if kw in k:
                    raise ValueError(f"Missing key: {k}")
                
            for kw in ["mask_estim", "query_encoder"]:
                if kw in k:
                    continue
            


    def adapt_query(self, q, batch):
        
        w = self.query_encoder(batch.query.audio)
        q = torch.permute(q, (0, 3, 1, 2)) # (batch, n_band, n_time, emb_dim) -> (batch, emb_dim, n_band, n_time)
        q = self.film(q, w)
        q = torch.permute(q, (0, 2, 3, 1)) # -> (batch, n_band, n_time, emb_dim)
        
        return q


    def optimized_forward(self, batch: InputType, mode: OperationMode = OperationMode.TRAIN):

        with torch.no_grad():
            x = self.stft(batch.mixture.audio)
            batch.mixture.spectrogram = x

            if "sources" in batch.keys():
                for stem in batch.sources.keys():
                    s = batch.sources[stem].audio
                    s = self.stft(s)
                    batch.sources[stem].spectrogram = s

        batch = self.optimized_separate(batch)

        return batch


    def optimized_separate(self, batch):

        x, q, length = self.encode(batch)

        for stem, query in batch.query.items():

            batch_ = SimpleishNamespace(**batch.__dict__)
            batch_.query = query

            q = self.adapt_query(q, batch_)

            m = self.mask_estim(q)
            s = self.mask(x, m)
            s = torch.reshape(s, x.shape)
            batch.estimates[stem] = SimpleishNamespace(
                audio=self.istft(s, length), spectrogram=s
            )

        return batch