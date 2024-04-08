import math
import os
import random
import warnings
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch_audiomentations.utils.object_dict import ObjectDict
import torchaudio as ta
from torch.utils import data
from tqdm import tqdm

from core.data.base import BaseSourceSeparationDataset
from core.types import input_dict

from . import clean_track_inst

from torch import Tensor, nn

DBFS_HOP_SIZE = int(0.125 * 44100)
DBFS_CHUNK_SIZE = int(1 * 44100)

INST_BY_OCCURRENCE = [
    "bass_guitar",
    "kick_drum",
    "snare_drum",
    "lead_male_singer",
    "distorted_electric_guitar",
    "clean_electric_guitar",
    "toms",
    "acoustic_guitar",
    "background_vocals",
    "hi_hat",
    "overheads",
    "atonal_percussion",
    "grand_piano",
    "cymbals",
    "lead_female_singer",
    "synth_lead",
    "bass_synthesizer",
    "synth_pad",
    "organ_electric_organ",
    "fx",
    "drum_machine",
    "string_section",
    "electric_piano",
    "full_acoustic_drumkit",
    "other_sounds",
    "pitched_percussion",
    "brass",
    "reeds",
    "contrabass_double_bass",
    "other_plucked",
    "other_strings",
    "other_wind",
    "cello",
    "other",
    "flutes",
    "viola_section",
    "viola",
    "cello_section",
]

FINE_LEVEL_INSTRUMENTS = {
    "lead_male_singer",
    "lead_female_singer",
    "human_choir",
    "background_vocals",
    "other_vocals",
    "bass_guitar",
    "bass_synthesizer",
    "contrabass_double_bass",
    "tuba",
    "bassoon",
    "snare_drum",
    "toms",
    "kick_drum",
    "cymbals",
    "overheads",
    "full_acoustic_drumkit",
    "drum_machine",
    "hihat",
    "fx",
    "click_track",
    "clean_electric_guitar",
    "distorted_electric_guitar",
    "lap_steel_guitar_or_slide_guitar",
    "acoustic_guitar",
    "other_plucked",
    "atonal_percussion",
    "pitched_percussion",
    "grand_piano",
    "electric_piano",
    "organ_electric_organ",
    "synth_pad",
    "synth_lead",
    "other_sounds",
    "violin",
    "viola",
    "cello",
    "violin_section",
    "viola_section",
    "cello_section",
    "string_section",
    "other_strings",
    "brass",
    "flutes",
    "reeds",
    "other_wind",
}

COARSE_LEVEL_INSTRUMENTS = {
    "vocals",
    "bass",
    "drums",
    "guitar",
    "other_plucked",
    "percussion",
    "piano",
    "other_keys",
    "bowed_strings",
    "wind",
    "other",
}

COARSE_TO_FINE = {
    "vocals": [
        "lead_male_singer",
        "lead_female_singer",
        "human_choir",
        "background_vocals",
        "other_vocals",
    ],
    "bass": [
        "bass_guitar",
        "bass_synthesizer",
        "contrabass_double_bass",
        "tuba",
        "bassoon",
    ],
    "drums": [
        "snare_drum",
        "toms",
        "kick_drum",
        "cymbals",
        "overheads",
        "full_acoustic_drumkit",
        "drum_machine",
        "hihat",
    ],
    "other": ["fx", "click_track"],
    "guitar": [
        "clean_electric_guitar",
        "distorted_electric_guitar",
        "lap_steel_guitar_or_slide_guitar",
        "acoustic_guitar",
    ],
    "other_plucked": ["other_plucked"],
    "percussion": ["atonal_percussion", "pitched_percussion"],
    "piano": ["grand_piano", "electric_piano"],
    "other_keys": ["organ_electric_organ", "synth_pad", "synth_lead", "other_sounds"],
    "bowed_strings": [
        "violin",
        "viola",
        "cello",
        "violin_section",
        "viola_section",
        "cello_section",
        "string_section",
        "other_strings",
    ],
    "wind": ["brass", "flutes", "reeds", "other_wind"],
}

COARSE_TO_FINE = {k: set(v) for k, v in COARSE_TO_FINE.items()}
FINE_TO_COARSE = {k: kk for kk, v in COARSE_TO_FINE.items() for k in v}

ALL_LEVEL_INSTRUMENTS = COARSE_LEVEL_INSTRUMENTS.union(FINE_LEVEL_INSTRUMENTS)


class MoisesDBBaseDataset(BaseSourceSeparationDataset, ABC):
    def __init__(
        self,
        split: str,
        data_path: str = "/home/kwatchar3/Documents/data/moisesdb",
        fs: int = 44100,
        return_stems: Union[bool, List[str]] = False,
        npy_memmap=True,
        recompute_mixture=False,
        train_folds=None,
        val_folds=None,
        test_folds=None,
        query_file="query",
    ) -> None:
        if test_folds is None:
            test_folds = [5]

        if val_folds is None:
            val_folds = [4]

        if train_folds is None:
            train_folds = [1, 2, 3]

        split_path = os.path.join(data_path, "splits.csv")
        splits = pd.read_csv(split_path)

        metadata_path = os.path.join(data_path, "stems.csv")
        metadata = pd.read_csv(metadata_path)

        if split == "train":
            folds = train_folds
        elif split == "val":
            folds = val_folds
        elif split == "test":
            folds = test_folds
        else:
            raise NameError

        files = splits[splits["split"].isin(folds)]["song_id"].tolist()
        metadata = metadata[metadata["song_id"].isin(files)]

        super().__init__(
            split=split,
            stems=["mixture"],
            files=files,
            data_path=data_path,
            fs=fs,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
        )

        self.folds = folds

        self.metadata = metadata.rename(
            columns={k: k.replace(" ", "_") for k in metadata.columns}
        )

        self.song_to_stem = (
            metadata.set_index("song_id")
            .apply(lambda row: row[row == 1].index.tolist(), axis=1)
            .to_dict()
        )
        self.stem_to_song = (
            metadata.set_index("song_id")
            .transpose()
            .apply(lambda row: row[row == 1].index.tolist(), axis=1)
            .to_dict()
        )

        self.true_length = len(self.files)
        self.n_channels = 2

        self.audio_path = os.path.join(data_path, "npy2")

        self.return_stems = return_stems

        self.query_file = query_file

    def get_full_stem(self, *, stem: str, identifier) -> torch.Tensor:
        song_id = identifier["song_id"]
        path = os.path.join(self.data_path, "npy2", song_id)
        # noinspection PyUnresolvedReferences

        assert self.npy_memmap

        if os.path.exists(os.path.join(path, f"{stem}.npy")):
            audio = np.load(os.path.join(path, f"{stem}.npy"), mmap_mode="r")
        else:
            audio = None

        return audio

    def get_query_stem(self, *, stem: str, identifier) -> torch.Tensor:
        song_id = identifier["song_id"]
        path = os.path.join(self.data_path, "npyq", song_id)
        # noinspection PyUnresolvedReferences

        if self.npy_memmap:
            # print(self.npy_memmap)
            audio = np.load(
                os.path.join(path, f"{stem}.{self.query_file}.npy"), mmap_mode="r"
            )
        else:
            raise NotImplementedError

        return audio

    def get_stem(self, *, stem: str, identifier) -> torch.Tensor:
        audio = self.get_full_stem(stem=stem, identifier=identifier)
        return audio

    def get_identifier(self, index):
        return dict(song_id=self.files[index % self.true_length])

    def __getitem__(self, index: int):
        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier)

        mixture = audio["mixture"].copy()

        if isinstance(self.return_stems, list):
            sources = {
                stem: audio.get(stem, np.zeros_like(mixture))
                for stem in self.return_stems
            }
        elif isinstance(self.return_stems, bool):
            if self.return_stems:
                sources = {
                    stem: audio[stem].copy()
                    for stem in self.song_to_stem[identifier["song_id"]]
                }
            else:
                sources = None
        else:
            raise ValueError

        return input_dict(
            mixture=mixture,
            sources=sources,
            metadata=identifier,
            modality="audio",
        )


class MoisesDBFullTrackDataset(MoisesDBBaseDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        return_stems: Union[bool, List[str]] = False,
        npy_memmap=True,
        recompute_mixture=False,
        query_file="query",
    ) -> None:
        super().__init__(
            split=split,
            data_path=data_root,
            return_stems=return_stems,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
            query_file=query_file,
        )

    def __len__(self) -> int:
        return self.true_length


class MoisesDBVDBOFullTrackDataset(MoisesDBFullTrackDataset):
    def __init__(
        self, data_root: str, split: str, npy_memmap=True, recompute_mixture=False
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            return_stems=["vocals", "bass", "drums", "vdbo_others"],
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
            query_file=None,
        )


import torch_audiomentations as audiomentations
from torch_audiomentations.utils.dsp import convert_decibels_to_amplitude_ratio


class SmartGain(audiomentations.Gain):
    def __init__(
        self, p=0.5, min_gain_in_db=-6, max_gain_in_db=6, dbfs_threshold=-45.0
    ):
        super().__init__(
            p=p, min_gain_in_db=min_gain_in_db, max_gain_in_db=max_gain_in_db
        )

        self.dbfs_threshold = dbfs_threshold

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):

        dbfs = 10 * torch.log10(torch.mean(torch.square(samples)) + 1e-6)

        if dbfs > self.dbfs_threshold:
            low = self.min_gain_in_db
        else:
            low = max(0.0, self.min_gain_in_db)

        distribution = torch.distributions.Uniform(
            low=torch.tensor(low, dtype=torch.float32, device=samples.device),
            high=torch.tensor(
                self.max_gain_in_db, dtype=torch.float32, device=samples.device
            ),
            validate_args=True,
        )
        selected_batch_size = samples.size(0)
        self.transform_parameters["gain_factors"] = (
            convert_decibels_to_amplitude_ratio(
                distribution.sample(sample_shape=(selected_batch_size,))
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )


class Audiomentations(audiomentations.Compose):
    def __init__(self, augment="gssp", fs: int = 44100):

        if isinstance(augment, str):
            if augment == "gssp":
                augment = OmegaConf.create(
                    [
                        dict(
                            cls="Shift",
                            kwargs=dict(p=1.0, min_shift=-0.5, max_shift=0.5),
                        ),
                        dict(
                            cls="Gain",
                            kwargs=dict(p=1.0, min_gain_in_db=-6, max_gain_in_db=6),
                        ),
                        dict(cls="ShuffleChannels", kwargs=dict(p=0.5)),
                        dict(cls="PolarityInversion", kwargs=dict(p=0.5)),
                    ]
                )
            else:
                raise ValueError

        transforms = []

        for transform in augment:

            if transform.cls == "Gain":
                transforms.append(SmartGain(**transform.kwargs))
            else:
                transforms.append(
                    getattr(audiomentations, transform.cls)(**transform.kwargs)
                )

        super().__init__(transforms=transforms, shuffle=True)

        self.fs = fs

    def forward(
        self,
        samples: torch.Tensor = None,
    ) -> ObjectDict:
        return super().forward(samples, sample_rate=self.fs)


class MoisesDBVDBORandomChunkDataset(MoisesDBVDBOFullTrackDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 4.0,
        fs: int = 44100,
        target_length: int = 8192,
        augment=None,
        npy_memmap=True,
        recompute_mixture=True,
        db_threshold=-24.0,
        db_step=-12.0,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
        )

        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = int(chunk_size_seconds * fs)
        self.fs = fs

        self.target_length = target_length

        self.db_threshold = db_threshold
        self.db_step = db_step

        if augment is not None:
            assert self.recompute_mixture
            self.augment = Audiomentations(augment, fs)
        else:
            self.augment = None

    def __len__(self) -> int:
        return self.target_length

    def _chunk_audio(self, audio, start, end):
        audio = {k: v[..., start:end] for k, v in audio.items()}

        return audio

    def _get_start_end(self, audio, identifier):
        n_samples = audio.shape[-1]
        start = np.random.randint(0, n_samples - self.chunk_size_samples)
        end = start + self.chunk_size_samples

        return start, end

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}

        for stem in stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        for stem in stems:
            if audio[stem] is None:
                audio[stem] = np.zeros(
                    audio[
                        (
                            "mixture"
                            if "mixture" in stems
                            else [s for s in stems if audio[s] is not None][0]
                        )
                    ].shape,
                    dtype=np.float32,
                )

        start, end = self._get_start_end(audio[stems[0]], identifier)
        audio = self._chunk_audio(audio, start, end)

        if self.augment is not None:
            audio = {
                k: self.augment(torch.from_numpy(v[None, :, :]))[0, :, :].numpy()
                for k, v in audio.items()
            }

        return audio

    def get_audio(self, identifier: Dict[str, Any]):
        if self.recompute_mixture:
            audio = self._get_audio(
                ["vocals", "bass", "drums", "vdbo_others"], identifier=identifier
            )
            audio["mixture"] = self.compute_mixture(audio)
            return audio
        else:
            return self._get_audio(
                ["mixture", "vocals", "bass", "drums", "vdbo_others"],
                identifier=identifier,
            )

    def __getitem__(self, index: int):

        identifier = self.get_identifier(index)
        audio = self.get_audio(identifier=identifier)

        mixture = audio["mixture"].copy()

        sources = {
            stem: audio.get(stem, np.zeros_like(mixture)) for stem in self.return_stems
        }

        return input_dict(
            mixture=mixture,
            sources=sources,
            metadata=identifier,
            modality="audio",
        )


class MoisesDBVDBODeterministicChunkDataset(MoisesDBVDBORandomChunkDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 4.0,
        hop_size_seconds: float = 8.0,
        fs: int = 44100,
        npy_memmap=True,
        recompute_mixture=False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            chunk_size_seconds=chunk_size_seconds,
            npy_memmap=npy_memmap,
            recompute_mixture=recompute_mixture,
        )

        self.hop_size_seconds = hop_size_seconds
        self.hop_size_samples = int(hop_size_seconds * fs)

        self.index_to_identifiers = self._generate_index()
        self.length = len(self.index_to_identifiers)

    def __len__(self) -> int:
        return self.length

    def _generate_index(self):

        identifiers = []

        for song_id in self.files:
            audio = self.get_full_stem(stem="mixture", identifier=dict(song_id=song_id))
            n_samples = audio.shape[-1]
            n_chunks = math.floor(
                (n_samples - self.chunk_size_samples) / self.hop_size_samples
            )

            for i in range(n_chunks):
                chunk_start = i * self.hop_size_samples
                identifiers.append(dict(song_id=song_id, chunk_start=chunk_start))

        return identifiers

    def get_identifier(self, index):
        return self.index_to_identifiers[index]

    def _get_start_end(self, audio, identifier):

        start = identifier["chunk_start"]
        end = start + self.chunk_size_samples

        return start, end


def round_samples(seconds, fs, hop_size, downsample):
    n_frames = math.ceil(seconds * fs / hop_size) + 1
    n_frames_down = math.ceil(n_frames / downsample)
    n_frames = n_frames_down * downsample
    n_samples = (n_frames - 1) * hop_size

    return int(n_samples)


class MoisesDBRandomChunkRandomQueryDataset(MoisesDBFullTrackDataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4.0,
        query_size_seconds: float = 1.0,
        round_query: bool = False,
        min_query_dbfs: float = -40.0,
        min_target_dbfs: float = -36.0,
        min_target_dbfs_step: float = -12.0,
        max_dbfs_tries: int = 10,
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap=True,
        allowed_stems=None,
        query_file="query",
        augment=None,
    ) -> None:

        super().__init__(
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            recompute_mixture=augment is not None,
            query_file=query_file,
        )

        self.mixture_stem = mixture_stem

        self.chunk_size_seconds = chunk_size_seconds
        self.chunk_size_samples = round_samples(
            self.chunk_size_seconds, self.fs, 512, 2**6
        )

        self.query_size_seconds = query_size_seconds

        if round_query:
            self.query_size_samples = round_samples(
                self.query_size_seconds, self.fs, 512, 2**6
            )
        else:
            self.query_size_samples = int(self.query_size_seconds * self.fs)

        self.target_length = target_length

        self.min_query_dbfs = min_query_dbfs

        if min_target_dbfs is None:
            min_target_dbfs = -np.inf
            min_target_dbfs_step = None
            max_dbfs_tries = 1

        self.min_target_dbfs = min_target_dbfs
        self.min_target_dbfs_step = min_target_dbfs_step
        self.max_dbfs_tries = max_dbfs_tries

        self.top_k_instrument = top_k_instrument

        if allowed_stems is None:
            allowed_stems = INST_BY_OCCURRENCE[: self.top_k_instrument]
        else:
            self.top_k_instrument = None

        self.allowed_stems = allowed_stems

        self.song_to_all_stems = {
            k: list(set(v) & set(ALL_LEVEL_INSTRUMENTS))
            for k, v in self.song_to_stem.items()
        }

        self.song_to_stem = {
            k: list(set(v) & set(self.allowed_stems))
            for k, v in self.song_to_stem.items()
        }
        self.stem_to_song = {
            k: list(set(v) & set(self.files)) for k, v in self.stem_to_song.items()
        }

        self.queriable_songs = [k for k, v in self.song_to_stem.items() if len(v) > 0]

        self.use_own_query = use_own_query

        if self.use_own_query:
            self.files = [k for k in self.files if len(self.song_to_stem[k]) > 0]
            self.true_length = len(self.files)

        if augment is not None:
            assert self.recompute_mixture
            self.augment = Audiomentations(augment, self.fs)
        else:
            self.augment = None

    def __len__(self) -> int:
        return self.target_length

    def _chunk_audio(self, audio, start, end):
        audio = {k: v[..., start:end] for k, v in audio.items()}

        return audio

    def _get_start_end(self, audio):
        n_samples = audio.shape[-1]
        start = np.random.randint(0, n_samples - self.chunk_size_samples)
        end = start + self.chunk_size_samples

        return start, end

    def _target_dbfs(self, audio):
        return 10.0 * np.log10(np.mean(np.square(np.abs(audio))) + 1e-6)

    def _chunk_and_check_dbfs_threshold(self, audio_, target_stem, threshold):

        target_dict = {target_stem: audio_[target_stem]}

        for _ in range(self.max_dbfs_tries):
            start, end = self._get_start_end(audio_[target_stem])
            taudio = self._chunk_audio(target_dict, start, end)

            dbfs = self._target_dbfs(taudio[target_stem])
            if dbfs > threshold:
                return self._chunk_audio(audio_, start, end)

        return None

    def _chunk_and_check_dbfs(self, audio_, target_stem):
        out = self._chunk_and_check_dbfs_threshold(
            audio_, target_stem, self.min_target_dbfs
        )

        if out is not None:
            return out

        out = self._chunk_and_check_dbfs_threshold(
            audio_, target_stem, self.min_target_dbfs + self.min_target_dbfs_step
        )

        if out is not None:
            return out

        start, end = self._get_start_end(audio_[target_stem])
        audio = self._chunk_audio(audio_, start, end)

        return audio

    def _augment(self, audio, target_stem):
        stack_audio = np.stack([v for v in audio.values()], axis=0)
        aug_audio = self.augment(torch.from_numpy(stack_audio)).numpy()
        mixture = np.sum(aug_audio, axis=0)

        out = {
            "mixture": mixture,
        }

        if target_stem is not None:
            target_idx = list(audio.keys()).index(target_stem)
            out[target_stem] = aug_audio[target_idx]

        return out

    def _choose_stems_for_augment(self, identifier, target_stem):
        stems_for_song = set(self.song_to_all_stems[identifier["song_id"]])

        stems_ = []
        coarse_level_accounted = set()

        is_none_target = target_stem is None
        is_coarse_target = target_stem in COARSE_LEVEL_INSTRUMENTS

        if is_coarse_target or is_none_target:
            coarse_target = target_stem
        else:
            coarse_target = FINE_TO_COARSE[target_stem]

        fine_level_stems = stems_for_song & FINE_LEVEL_INSTRUMENTS
        coarse_level_stems = stems_for_song & COARSE_LEVEL_INSTRUMENTS

        for s in fine_level_stems:
            coarse_level = FINE_TO_COARSE[s]

            if is_coarse_target and coarse_level == coarse_target:
                continue
            else:
                stems_.append(s)

            coarse_level_accounted.add(coarse_level)

        stems_ += list(coarse_level_stems - coarse_level_accounted)

        if target_stem is not None:
            assert target_stem in stems_, f"stems: {stems_}, target stem: {target_stem}"

            if len(stems_for_song) > 1:
                assert (
                    len(stems_) > 1
                ), f"stems: {stems_}, stems in song: {stems_for_song},\n target stem: {target_stem}"

        assert "mixture" not in stems_

        return stems_

    def _get_audio(
        self, stems, identifier: Dict[str, Any], check_dbfs=True, no_target=False
    ):

        target_stem = stems[0] if not no_target else None

        if self.augment is not None:
            stems_ = self._choose_stems_for_augment(identifier, target_stem)
        else:
            stems_ = stems

        audio = {}
        for stem in stems_:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        audio_ = {k: v.copy() for k, v in audio.items()}

        if check_dbfs:
            assert target_stem is not None
            audio = self._chunk_and_check_dbfs(audio_, target_stem)
        else:
            first_key = list(audio_.keys())[0]
            start, end = self._get_start_end(audio_[first_key])
            audio = self._chunk_audio(audio_, start, end)

        if self.augment is not None:
            assert "mixture" not in audio
            audio = self._augment(audio, target_stem)
            assert "mixture" in audio

        return audio

    def __getitem__(self, index: int):

        mix_identifier = self.get_identifier(index)
        mix_stems = self.song_to_stem[mix_identifier["song_id"]]

        if self.use_own_query:
            query_id = mix_identifier["song_id"]
            query_identifier = dict(song_id=query_id)
            possible_stem = mix_stems

            assert len(possible_stem) > 0

            zero_target = False
        else:
            query_id = random.choice(self.queriable_songs)
            query_identifier = dict(song_id=query_id)
            query_stems = self.song_to_stem[query_id]
            possible_stem = list(set(mix_stems) & set(query_stems))

            if len(possible_stem) == 0:
                possible_stem = query_stems
                zero_target = True
                # print(f"Mix {mix_identifier['song_id']} and query {query_id} have no common stems.")
                # return self.__getitem__(index + 1)
            else:
                zero_target = False

        assert (
            len(possible_stem) > 0
        ), f"{mix_identifier['song_id']} and {query_id} have no common stems. zero target is {zero_target}"
        stem = random.choice(possible_stem)

        if zero_target:
            audio = self._get_audio(
                [self.mixture_stem],
                identifier=mix_identifier,
                check_dbfs=False,
                no_target=True,
            )
            mixture = audio[self.mixture_stem].copy()
            sources = {"target": np.zeros_like(mixture)}
        else:
            audio = self._get_audio(
                [stem, self.mixture_stem], identifier=mix_identifier, check_dbfs=True
            )
            mixture = audio[self.mixture_stem].copy()
            sources = {"target": audio[stem].copy()}

        query = self.get_query_stem(stem=stem, identifier=query_identifier)
        query = query.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBRandomChunkBalancedRandomQueryDataset(
    MoisesDBRandomChunkRandomQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        target_length: int,
        chunk_size_seconds: float = 4,
        query_size_seconds: float = 1,
        round_query: bool = False,
        min_query_dbfs: float = -40.0,
        min_target_dbfs: float = -36.0,
        min_target_dbfs_step: float = -12.0,
        max_dbfs_tries: int = 10,
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap=True,
        allowed_stems=None,
        query_file="query",
        augment=None,
    ) -> None:
        super().__init__(
            data_root,
            split,
            target_length,
            chunk_size_seconds,
            query_size_seconds,
            round_query,
            min_query_dbfs,
            min_target_dbfs,
            min_target_dbfs_step,
            max_dbfs_tries,
            top_k_instrument,
            mixture_stem,
            use_own_query,
            npy_memmap,
            allowed_stems,
            query_file,
            augment,
        )
        
        self.stem_to_n_songs = {k: len(v) for k, v in self.stem_to_song.items()}
        self.trainable_stems = [k for k, v in self.stem_to_n_songs.items() if v > 1]
        self.n_allowed_stems = len(self.allowed_stems)
        
        
        
    def __getitem__(self, index: int):
        
        stem = self.allowed_stems[index % self.n_allowed_stems]
        song_ids_with_stem = self.stem_to_song[stem]
        
        song_id = song_ids_with_stem[index % self.stem_to_n_songs[stem]]
        
        mix_identifier = dict(song_id=song_id)
        
        audio = self._get_audio([stem, self.mixture_stem], identifier=mix_identifier, check_dbfs=True)
        mixture = audio[self.mixture_stem].copy()
        
        if self.use_own_query:
            query_id = song_id
            query_identifier = dict(song_id=query_id)
        else:
            query_id = random.choice(song_ids_with_stem)
            query_identifier = dict(song_id=query_id)
            
        query = self.get_query_stem(stem=stem, identifier=query_identifier)
        query = query.copy()
        
        sources = {"target": audio[stem].copy()}
        
        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )
        
        


class MoisesDBDeterministicChunkDeterministicQueryDataset(
    MoisesDBRandomChunkRandomQueryDataset
):
    def __init__(
        self,
        data_root: str,
        split: str,
        chunk_size_seconds: float = 4.0,
        hop_size_seconds: float = 8.0,
        query_size_seconds: float = 1.0,
        min_query_dbfs: float = -40.0,
        top_k_instrument: int = 10,
        n_queries_per_chunk: int = 1,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap=True,
        allowed_stems: List[str] = None,
        query_file="query",
    ) -> None:

        super().__init__(
            data_root=data_root,
            split=split,
            target_length=None,
            chunk_size_seconds=chunk_size_seconds,
            query_size_seconds=query_size_seconds,
            min_query_dbfs=min_query_dbfs,
            top_k_instrument=top_k_instrument,
            mixture_stem=mixture_stem,
            use_own_query=use_own_query,
            npy_memmap=npy_memmap,
            allowed_stems=allowed_stems,
            query_file=query_file,
        )

        if hop_size_seconds is None:
            hop_size_seconds = chunk_size_seconds

        self.chunk_hop_size_seconds = hop_size_seconds

        self.chunk_hop_size_samples = int(hop_size_seconds * self.fs)

        self.n_queries_per_chunk = n_queries_per_chunk

        self._overwrite = False

        self.query_tuples = self.find_query_tuples_or_generate()
        self.n_chunks = len(self.query_tuples)

    def __len__(self) -> int:
        return self.n_chunks

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}

        for stem in stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        start = identifier["chunk_start"]
        end = start + self.chunk_size_samples
        audio = self._chunk_audio(audio, start, end)

        return audio

    def find_query_tuples_or_generate(self):
        query_path = os.path.join(self.data_path, "queries")
        val_folds = "-".join(map(str, self.folds))

        path_so_far = os.path.join(query_path, val_folds)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        chunk_specs = f"chunk{self.chunk_size_samples}-hop{self.chunk_hop_size_samples}"
        path_so_far = os.path.join(path_so_far, chunk_specs)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        query_specs = f"query{self.query_size_samples}-n{self.n_queries_per_chunk}"
        path_so_far = os.path.join(path_so_far, query_specs)

        if not os.path.exists(path_so_far):
            return self.generate_index()

        if self.top_k_instrument is not None:
            path_so_far = os.path.join(
                path_so_far, f"queries-top{self.top_k_instrument}.csv"
            )
        else:
            if len(self.allowed_stems) > 5:
                allowed_stems = (
                    str(len(self.allowed_stems))
                    + "stems:"
                    + ":".join([k[0] for k in self.allowed_stems if k != "mixture"])
                )
            else:
                allowed_stems = ":".join(self.allowed_stems)

            path_so_far = os.path.join(path_so_far, f"queries-{allowed_stems}.csv")

        if not os.path.exists(path_so_far):
            return self.generate_index()

        print(f"Loading query tuples from {path_so_far}")

        return pd.read_csv(path_so_far)

    def _get_index_path(self):
        query_root = os.path.join(self.data_path, "queries")
        val_folds = "-".join(map(str, self.folds))
        chunk_specs = f"chunk{self.chunk_size_samples}-hop{self.chunk_hop_size_samples}"
        query_specs = f"query{self.query_size_samples}-n{self.n_queries_per_chunk}"
        query_dir = os.path.join(query_root, val_folds, chunk_specs, query_specs)

        if self.top_k_instrument is not None:
            query_path = os.path.join(
                query_dir, f"queries-top{self.top_k_instrument}.csv"
            )
        else:
            if len(self.allowed_stems) > 5:
                allowed_stems = (
                    str(len(self.allowed_stems))
                    + "stems:"
                    + ":".join([k[0] for k in self.allowed_stems if k != "mixture"])
                )
            else:
                allowed_stems = ":".join(self.allowed_stems)
            query_path = os.path.join(query_dir, f"queries-{allowed_stems}.csv")

        if not self._overwrite:
            assert not os.path.exists(
                query_path
            ), f"Query path {query_path} already exists."

        os.makedirs(query_dir, exist_ok=True)

        return query_path

    def generate_index(self):

        query_path = self._get_index_path()

        durations = pd.read_csv(os.path.join(self.data_path, "durations.csv"))
        durations = (
            durations[["song_id", "duration"]]
            .set_index("song_id")["duration"]
            .to_dict()
        )

        tuples = []

        stems_without_queries = defaultdict(list)

        for i, song_id in tqdm(enumerate(self.files), total=len(self.files)):
            song_duration = durations[song_id]
            mix_stems = self.song_to_stem[song_id]

            n_mix_chunks = math.floor(
                (song_duration - self.chunk_size_seconds) / self.chunk_hop_size_seconds
            )

            for stem in mix_stems:
                possible_queries = self.stem_to_song[stem]
                if song_id in possible_queries:
                    possible_queries.remove(song_id)

                if len(possible_queries) == 0:
                    stems_without_queries[song_id].append(stem)
                    continue

                for k in tqdm(range(n_mix_chunks), desc=f"song{i + 1}/{stem}"):
                    mix_chunk_start = int(k * self.chunk_hop_size_samples)

                    for j in range(self.n_queries_per_chunk):
                        query = random.choice(possible_queries)

                        tuples.append(
                            dict(
                                mix=song_id,
                                query=query,
                                stem=stem,
                                mix_chunk_start=mix_chunk_start,
                            )
                        )

        if len(stems_without_queries) > 0:
            print("Stems without queries:")
            for song_id, stems in stems_without_queries.items():
                print(f"{song_id}: {stems}")

        tuples = pd.DataFrame(tuples)

        print(
            f"Generating query tuples for {self.split} set with {len(tuples)} tuples."
        )
        print(f"Saving query tuples to {query_path}")

        tuples.to_csv(query_path, index=False)

        return tuples

    def index_to_identifiers(self, index: int) -> Tuple[str, str, str, int]:

        row = self.query_tuples.iloc[index]
        mix_id = row["mix"]

        if self.use_own_query:
            query_id = mix_id
        else:
            query_id = row["query"]

        stem = row["stem"]
        mix_chunk_start = row["mix_chunk_start"]

        return mix_id, query_id, stem, mix_chunk_start

    def __getitem__(self, index: int):

        mix_id, query_id, stem, mix_chunk_start = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id, chunk_start=mix_chunk_start)
        query_identifier = dict(song_id=query_id)

        audio = self._get_audio([stem, self.mixture_stem], identifier=mix_identifier)
        query = self.get_query_stem(stem=stem, identifier=query_identifier)

        mixture = audio[self.mixture_stem].copy()
        sources = {"target": audio[stem].copy()}
        query = query.copy()

        assert mixture.shape[-1] == self.chunk_size_samples
        # print(query.shape[-1], self.query_size_samples)
        assert query.shape[-1] == self.query_size_samples
        assert sources["target"].shape[-1] == self.chunk_size_samples

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier,
                "query": query_identifier,
                "stem": stem,
            },
            modality="audio",
        )


class MoisesDBFullTrackTestQueryDataset(MoisesDBFullTrackDataset):
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        top_k_instrument: int = 10,
        mixture_stem: str = "mixture",
        use_own_query: bool = True,
        npy_memmap=True,
        allowed_stems: List[str] = None,
        query_file="query-10s",
    ) -> None:
        super().__init__(
            data_root=data_root,
            split=split,
            npy_memmap=npy_memmap,
            recompute_mixture=False,
            query_file=query_file,
        )

        self.use_own_query = use_own_query

        self.allowed_stems = allowed_stems

        test_indices = pd.read_csv(os.path.join(data_root, "test_indices.csv"))

        test_indices = test_indices[test_indices.stem.isin(self.allowed_stems)]

        self.test_indices = test_indices

        self.length = len(self.test_indices)

    def __len__(self) -> int:
        return self.length

    def index_to_identifiers(self, index: int) -> Tuple[str, str, str]:

        row = self.test_indices.iloc[index]
        mix_id = row["song_id"]
        if self.use_own_query:
            query_id = mix_id
        else:
            query_id = row["query_id"]
        stem = row["stem"]

        return mix_id, query_id, stem

    def _get_audio(self, stems, identifier: Dict[str, Any]):
        audio = {}

        for stem in stems:
            audio[stem] = self.get_full_stem(stem=stem, identifier=identifier)

        return audio

    def __getitem__(self, index: int):

        mix_id, query_id, stem = self.index_to_identifiers(index)

        mix_identifier = dict(song_id=mix_id)

        query_identifier = dict(song_id=query_id)

        audio = self._get_audio([stem, "mixture"], identifier=mix_identifier)
        query = self.get_query_stem(stem=stem, identifier=query_identifier)

        mixture = audio["mixture"].copy()
        sources = {stem: audio[stem].copy()}
        query = query.copy()

        return input_dict(
            mixture=mixture,
            sources=sources,
            query=query,
            metadata={
                "mix": mix_identifier["song_id"],
                "query": query_identifier["song_id"],
                "stem": stem,
            },
            modality="audio",
        )


if __name__ == "__main__":

    print("Beginning")

    config = "/storage/home/hcoda1/1/kwatchar3/coda/config/data/moisesdb-everything-query-d-aug.yml"

    config = OmegaConf.load(config)

    print("Loaded config")

    dataset = MoisesDBRandomChunkRandomQueryDataset(
        data_root=config.data_root, split="train", **config.train_kwargs
    )

    print("Loaded dataset")

    for item in tqdm(dataset, total=len(dataset)):
        target_audio = item["sources"]["target"]["audio"]
        mixture = item["mixture"]["audio"]

        if target_audio is None:
            raise ValueError
        else:
            tdb = 10.0 * torch.log10(torch.mean(torch.square(target_audio)) + 1e-6)
            mdb = 10.0 * torch.log10(torch.mean(torch.square(mixture)) + 1e-6)
            print(f"Target db: {tdb}, Mixture db: {mdb}")
