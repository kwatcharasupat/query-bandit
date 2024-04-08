from collections import defaultdict
import glob
import json
import math
import os
import shutil
from itertools import chain
from pprint import pprint
from types import SimpleNamespace
import numpy as np
import pandas as pd

from omegaconf import OmegaConf

from tqdm.contrib.concurrent import process_map

from tqdm import tqdm as tdqm, tqdm
import torchaudio as ta

import librosa

taxonomy = {
    "vocals": [
        "lead male singer",
        "lead female singer",
        "human choir",
        "background vocals",
        "other (vocoder, beatboxing etc)",
    ],
    "bass": [
        "bass guitar",
        "bass synthesizer (moog etc)",
        "contrabass/double bass (bass of instrings)",
        "tuba (bass of brass)",
        "bassoon (bass of woodwind)",
    ],
    "drums": [
        "snare drum",
        "toms",
        "kick drum",
        "cymbals",
        "overheads",
        "full acoustic drumkit",
        "drum machine",
        "hi-hat"
    ],
    "other": [
        "fx/processed sound, scratches, gun shots, explosions etc",
        "click track",
    ],
    "guitar": [
        "clean electric guitar",
        "distorted electric guitar",
        "lap steel guitar or slide guitar",
        "acoustic guitar",
    ],
    "other plucked": ["banjo, mandolin, ukulele, harp etc"],
    "percussion": [
        "a-tonal percussion (claps, shakers, congas, cowbell etc)",
        "pitched percussion (mallets, glockenspiel, ...)",
    ],
    "piano": [
        "grand piano",
        "electric piano (rhodes, wurlitzer, piano sound alike)",
    ],
    "other keys": [
        "organ, electric organ",
        "synth pad",
        "synth lead",
        "other sounds (hapischord, melotron etc)",
    ],
    "bowed strings": [
        "violin (solo)",
        "viola (solo)",
        "cello (solo)",
        "violin section",
        "viola section",
        "cello section",
        "string section",
        "other strings",
    ],
    "wind": [
        "brass (trumpet, trombone, french horn, brass etc)",
        "flutes (piccolo, bamboo flute, panpipes, flutes etc)",
        "reeds (saxophone, clarinets, oboe, english horn, bagpipe)",
        "other wind",
    ],
}

def clean_npy_other_vox(data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npyq"):
    npys = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)
    
    
    npys = [npy for npy in npys if "other" in npy]
    npys = [npy for npy in npys if "vdbo_" not in npy]
    npys = [npy for npy in npys if "other_" not in npy]

    stems = set([
        os.path.basename(npy).split(".")[0] for npy in npys
    ])
    
    assert len(stems) == 1
    
    for npy in tqdm(npys):
        shutil.move(npy, npy.replace("other", "other_vocals"))
    
    


def clean_track_inst(inst):
    
    if "vocoder" in inst:
        inst = "other_vocals"

    if "fx" in inst:
        inst = "fx"

    if "contrabass_double_bass" in inst:
        inst = "double_bass"

    if "banjo" in inst:
        return "other_plucked"

    if "(" in inst:
        inst = inst.split("(")[0]

    for s in [",", "-"]:
        if s in inst:
            inst = inst.replace(s, "")

    for s in ["/"]:
        if s in inst:
            inst = inst.replace(s, "_")

    if inst[-1] == "_":
        inst = inst[:-1]

    return inst


taxonomy = {
    k.replace(" ", "_"): [clean_track_inst(i.replace(" ", "_")) for i in v] for k, v in taxonomy.items()
}

fine_to_coarse = {}

for k, v in taxonomy.items():
    for vv in v:
        fine_to_coarse[vv] = k

# pprint(fine_to_coarse)

def save_taxonomy():
    with open("taxonomy.json", "w") as f:
        json.dump(taxonomy, f, indent=4)

    taxonomy_coarse = list(taxonomy.keys())
    
    with open("taxonomy_coarse.json", "w") as f:
        json.dump(taxonomy_coarse, f, indent=4)
        
    taxonomy_fine = list(chain(*taxonomy.values()))
    
    count_ = defaultdict(int)
    for t in taxonomy_fine:
        count_[t] += 1
        
    with open("taxonomy_fine.json", "w") as f:
        json.dump(taxonomy_fine, f, indent=4)
    


possible_coarse = list(taxonomy.keys())
possible_fine = list(set(chain(*taxonomy.values())))


def trim_and_mix(audios, length_=None):
    length = min([a.shape[-1] for a in audios])
    
    if length_ is not None:
        length = min(length, length_)
    
    audios = [a[..., :length] for a in audios]
    return np.sum(np.stack(audios, axis=0), axis=0), length


def retrim_npys(saved_npy, new_length):
    print("retrimming")
    for npy in saved_npy:
        audio = np.load(npy)
        audio = audio[..., :new_length]
        np.save(npy, audio)


def convert_one(inout):
    input_path = inout.input_path
    output_root = inout.output_root

    song_id = os.path.basename(input_path)
    output_root = os.path.join(output_root, song_id)
    os.makedirs(output_root, exist_ok=True)

    metadata = OmegaConf.load(os.path.join(input_path, "data.json"))
    stems = metadata.stems

    min_length = None
    saved_npy = []

    all_tracks = []
    other_tracks = []
    
    outfile = None
    
    added_tracks = set()
    duplicated_tracks = set()
    track_to_stem = defaultdict(list)
    added_stems = set()
    duplicated_stems = set()
    
    stem_name_to_stems = defaultdict(list)
    
    for stem in stems:
        stem_name = stem.stemName
        stem_name_to_stems[stem_name].append(stem)
    
        
    for stem_name in tqdm(stem_name_to_stems):
        stem_tracks = []
        for stem in stem_name_to_stems[stem_name]:
            stem_name = stem.stemName
            
            if stem_name in added_stems:
                print(f"Duplicate stem {stem_name} in {song_id}")
                duplicated_stems.add(stem_name)
            
            added_stems.add(stem_name)
            
            for track in stem.tracks:
                track_inst = track.trackType
                track_inst = clean_track_inst(track_inst)
                
                if track_inst in added_tracks:
                    if stem_name in track_to_stem[track_inst]:
                        continue
                    print(f"Duplicate track {track_inst} in {song_id}")
                    print(f"Stems: {track_to_stem[track_inst]}")
                    duplicated_tracks.add(track_inst)
                    raise ValueError
                else:
                    added_tracks.add(track_inst)
                    
                track_to_stem[track_inst].append(stem_name)
                track_id = track.id
                
                audio, fs = ta.load(os.path.join(input_path, stem_name, f"{track_id}.wav"))

                if fs != 44100:
                    print(f"fs is {fs} for {track_id}")
                    with open(os.path.join(output_root, "fs.txt"), "w") as f:
                        f.write(f"{song_id}\t{track_id}\t{fs}\n")

                if min_length is None:
                    min_length = audio.shape[-1]
                else:
                    if audio.shape[-1] < min_length:
                        min_length = audio.shape[-1]

                        if len(saved_npy) > 0:
                            retrim_npys(saved_npy, min_length)

                audio = audio[..., :min_length]
                audio = audio.numpy()
                audio = audio.astype(np.float32)

                if audio.shape[0] == 1:
                    print("mono")
                if audio.shape[0] > 2:
                    print("multi channel")

                assert outfile is None
                outfile = os.path.join(output_root, f"{track_inst}.npy")
                np.save(outfile, audio)
                saved_npy.append(outfile)
                outfile = None
                stem_tracks.append(audio)
                audio = None
                
        stem_track, min_length = trim_and_mix(stem_tracks)

        assert outfile is None
        outfile = os.path.join(output_root, f"{stem_name}.npy")
        np.save(outfile, stem_track)
        saved_npy.append(outfile)
        outfile = None
        
        all_tracks.append(stem_track)
        
        if stem_name not in ["vocals", "drums", "bass"]:
            # print(f"Putting {stem_name} in other")
            other_tracks.append(stem_track)
            
        
    assert outfile is None
    all_track, min_length_ = trim_and_mix(all_tracks, min_length)
    outfile = os.path.join(output_root, f"mixture.npy")
    np.save(outfile, all_track)
    
    if min_length_ != min_length:
        retrim_npys(saved_npy, min_length_)
        min_length = min_length_
    
    saved_npy.append(outfile)
    outfile = None
    
    other_track, min_length_ = trim_and_mix(other_tracks, min_length)
    np.save(os.path.join(output_root, f"vdbo_others.npy"), other_track)
    
    if min_length_ != min_length:
        retrim_npys(saved_npy, min_length_)
        min_length = min_length_


def convert_to_npy(
    data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/canonical",
    output_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npy2",
):
    if output_root is None:
        output_root = os.path.join(os.path.dirname(data_root), "npy")

    files = os.listdir(data_root)
    files = [
        os.path.join(data_root, f)
        for f in files
        if os.path.isdir(os.path.join(data_root, f))
    ]

    inout = [SimpleNamespace(input_path=f, output_root=output_root) for f in files]

    process_map(convert_one, inout)

    # for io in tdqm(inout):
    #     convert_one(io)


def make_others_one(input_path, dry_run=False):

    other_stems = [k for k in taxonomy.keys() if k not in ["vocals", "bass", "drums"]]
    npys = glob.glob(os.path.join(input_path, "**/*.npy"), recursive=True)

    npys = [npy for npy in npys if ".dbfs" not in npy]
    npys = [npy for npy in npys if ".query" not in npy]
    npys = [npy for npy in npys if "mixture" not in npy]
    npys = [npy for npy in npys if os.path.basename(npy).split(".")[0] in other_stems]

    print(f"Using stems: {[os.path.basename(npy).split('.')[0] for npy in npys]}")

    if len(npys) == 0:
        audio = np.zeros_like(np.load(os.path.join(input_path, "mixture.npy")))
    else:
        audio = [np.load(npy) for npy in npys]

        audio = np.sum(np.stack(audio, axis=0), axis=0)
    assert audio.shape[0] == 2

    output = os.path.join(input_path, "vdbo_others.npy")

    if dry_run:
        return

    np.save(output, audio)


def check_vdbo_one(f):
    s = np.sum(
        np.stack(
            [
                np.load(os.path.join(f, s + ".npy"))
                for s in ["vocals", "drums", "bass", "vdbo_others"]
                if os.path.exists(os.path.join(f, s + ".npy"))
            ],
            axis=0,
        ),
        axis=0,
    )
    m = np.load(os.path.join(f, "mixture.npy"))
    snr = 10 * np.log10(np.mean(np.square(m)) / np.mean(np.square(s - m)))
    print(snr)
    
    return snr

def check_vdbo(data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npy2"):
    files = os.listdir(data_root)

    files = [
        os.path.join(data_root, f)
        for f in files
        if os.path.isdir(os.path.join(data_root, f))
    ]

    snrs = process_map(check_vdbo_one, files)

    np.save("/storage/home/hcoda1/1/kwatchar3/data/vdbo.npy", np.array(snrs))


def make_others(data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npy2"):

    files = os.listdir(data_root)

    files = [
        os.path.join(data_root, f)
        for f in files
        if os.path.isdir(os.path.join(data_root, f))
    ]

    process_map(make_others_one, files)

    # for f in tqdm(files):
    #     make_others_one(f, dry_run=False)


def extract_metadata_one(input_path):
    song_id = os.path.basename(input_path)
    metadata = OmegaConf.load(os.path.join(input_path, "data.json"))

    song = metadata.song
    artist = metadata.artist
    genre = metadata.genre

    stems = metadata.stems
    data_out = []

    for stem in stems:
        stem_name = stem.stemName
        stem_id = stem.id
        for track in stem.tracks:
            track_inst = track.trackType
            track_id = track.id

            data_out.append(
                {
                    "song_id": song_id,
                    "song": song,
                    "artist": artist,
                    "genre": genre,
                    "stem_name": stem_name,
                    "stem_id": stem_id,
                    "track_inst": track_inst,
                    "track_id": track_id,
                    "has_bleed": track.has_bleed,
                }
            )

    return data_out


def consolidate_metadata(
    data_root="/home/kwatchar3/Documents/data/moisesdb/canonical",
):

    files = os.listdir(data_root)
    files = [
        os.path.join(data_root, f)
        for f in files
        if os.path.isdir(os.path.join(data_root, f))
    ]

    data = process_map(extract_metadata_one, files)

    df = pd.DataFrame.from_records(list(chain(*data)))

    df.to_csv(os.path.join(os.path.dirname(data_root), "metadata.csv"), index=False)


def clean_canonical(data_root="/home/kwatchar3/Documents/data/moisesdb/canonical"):

    npy = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)

    for n in tqdm(npy):
        os.remove(n)


def remove_dbfs(data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npy"):
    npy = glob.glob(os.path.join(data_root, "**/*.dbfs.npy"), recursive=True)

    for n in tqdm(npy):
        os.remove(n)


def make_split(
    metadata_path="/home/kwatchar3/Documents/data/moisesdb/metadata.csv",
    n_splits=5,
    seed=42,
):

    df = pd.read_csv(metadata_path)
    # print(df.columns)
    df = df[["song_id", "genre"]].drop_duplicates()

    genres = df["genre"].value_counts()
    genres_map = {g: g if c > n_splits else "other" for g, c in genres.items()}

    df["genre"] = df["genre"].map(genres_map)

    n_samples = len(df)
    n_per_split = n_samples // n_splits

    np.random.seed(seed)

    from sklearn.model_selection import train_test_split

    splits = []

    df_ = df.copy()

    for i in range(n_splits - 1):
        df_, test = train_test_split(
            df_,
            test_size=n_per_split,
            random_state=seed,
            stratify=df_["genre"],
            shuffle=True,
        )

        dfs = test[["song_id"]].copy().sort_values(by="song_id")
        dfs["split"] = i + 1
        splits.append(dfs)

    test = df_
    dfs = test[["song_id"]].copy().sort_values(by="song_id")
    dfs["split"] = n_splits
    splits.append(dfs)

    splits = pd.concat(splits)

    splits.to_csv(
        os.path.join(os.path.dirname(metadata_path), "splits.csv"), index=False
    )


def consolidate_stems(data_root="/home/kwatchar3/Documents/data/moisesdb/npy"):

    metadata = pd.read_csv(os.path.join(os.path.dirname(data_root), "metadata.csv"))

    dfg = metadata.groupby("song_id")[["stem_name", "track_inst"]]

    pprint(dfg)

    df = []

    def make_stem_dict(song_id, track_inst, stem_names):

        d = {"song_id": song_id}

        for inst in possible_fine:
            d[inst] = int(inst in track_inst)

        for inst in possible_coarse:
            d[inst] = int(inst in stem_names)

        return d

    for song_id, dfgg in dfg:

        track_inst = dfgg["track_inst"].tolist()
        track_inst = list(set(track_inst))
        track_inst = [clean_track_inst(inst) for inst in track_inst]

        stem_names = dfgg["stem_name"].tolist()
        stem_names = list(set([clean_track_inst(inst) for inst in stem_names]))

        d = make_stem_dict(song_id, track_inst, stem_names)
        df.append(d)

    print(df)

    df = pd.DataFrame.from_records(df)

    df.to_csv(os.path.join(os.path.dirname(data_root), "stems.csv"), index=False)


def get_dbfs(data_root="/home/kwatchar3/Documents/data/moisesdb/npy"):
    npys = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)

    dbfs = []

    for npy in tqdm(npys):
        audio = np.load(npy)
        song_id = os.path.basename(os.path.dirname(npy))
        track_id = os.path.basename(npy).split(".")[0]

        dbfs.append(
            {
                "song_id": song_id,
                "track_id": track_id,
                "dbfs": 10 * np.log10(np.mean(np.square(audio))),
            }
        )

    dbfs = pd.DataFrame.from_records(dbfs)

    dbfs.to_csv(os.path.join(os.path.dirname(data_root), "dbfs.csv"), index=False)

    return dbfs


def get_dbfs_by_chunk_one(inout):

    audio = np.load(inout.audio_path, mmap_mode="r")
    chunk_size = inout.chunk_size
    fs = inout.fs
    hop_size = inout.hop_size

    n_chan, n_samples = audio.shape
    chunk_size_samples = int(chunk_size * fs)
    hop_size_samples = int(hop_size * fs)

    x2win = np.lib.stride_tricks.sliding_window_view(
        np.square(audio), chunk_size_samples, axis=1
    )[:, ::hop_size_samples, :]

    x2win_mean = np.mean(x2win, axis=(0, 2))
    x2win_mean[x2win_mean == 0] = 1e-8
    dbfs = 10 * np.log10(x2win_mean)

    # song_id = os.path.basename(os.path.dirname(inout.audio_path))
    track_id = os.path.basename(inout.audio_path).split(".")[0]

    np.save(
        os.path.join(os.path.dirname(inout.audio_path), f"{track_id}.dbfs.npy"), dbfs
    )


def clean_data_root(data_root="/home/kwatchar3/Documents/data/moisesdb/npy"):
    npys = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)

    for npy in tqdm(npys):
        if ".dbfs" in npy or ".query" in npy:
            # print("removing", npy)
            os.remove(npy)


#
def get_dbfs_by_chunk(
    data_root="/home/kwatchar3/Documents/data/moisesdb/npy",
    query_root="/home/kwatchar3/Documents/data/moisesdb/npyq",
):
    npys = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)

    inout = [
        SimpleNamespace(
            audio_path=npy,
            chunk_size=1,
            hop_size=0.125,
            fs=44100,
            output_path=npy.replace(data_root, query_root).replace(
                ".npy", ".query.npy"
            ),
        )
        for npy in npys
    ]

    process_map(get_dbfs_by_chunk_one, inout, chunksize=2)


def round_samples(seconds, fs, hop_size, downsample):
    n_frames = math.ceil(seconds * fs / hop_size) + 1
    n_frames_down = math.ceil(n_frames / downsample)
    n_frames = n_frames_down * downsample
    n_samples = (n_frames - 1) * hop_size

    return int(n_samples)


def get_query_one(inout):

    audio = np.load(inout.audio_path, mmap_mode="r")
    chunk_size = inout.chunk_size
    fs = inout.fs
    output_path = inout.output_path
    round = inout.round
    hop_size = inout.hop_size

    if round:
        chunk_size_samples = round_samples(chunk_size, fs, 512, 2**6)
    else:
        chunk_size_samples = int(chunk_size * fs)

    audio_mono = np.mean(audio, axis=0)

    onset = librosa.onset.onset_detect(
        y=audio_mono, sr=fs, units="frames", hop_length=hop_size
    )

    onset_strength = librosa.onset.onset_strength(
        y=audio_mono, sr=fs, hop_length=hop_size
    )

    n_frames_per_chunk = chunk_size_samples // hop_size

    onset_strength_slide = np.lib.stride_tricks.sliding_window_view(
        onset_strength, n_frames_per_chunk, axis=0
    )

    onset_strength = np.mean(onset_strength_slide, axis=1)

    max_onset_frame = np.argmax(onset_strength)

    max_onset_samples = librosa.frames_to_samples(max_onset_frame)

    track_id = os.path.basename(inout.audio_path).split(".")[0]

    segment = audio[:, max_onset_samples : max_onset_samples + chunk_size_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, segment)


def get_query_from_onset(
    data_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npy2",  # "/home/kwatchar3/Documents/data/moisesdb/npy",
    query_root="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/npyq",  # "/home/kwatchar3/Documents/data/moisesdb/npyq",
    query_file="query-10s",
    pmap=True,
):
    npys = glob.glob(os.path.join(data_root, "**/*.npy"), recursive=True)

    npys = [npy for npy in npys if "dbfs" not in npy]

    inout = [
        SimpleNamespace(
            audio_path=npy,
            chunk_size=10,
            hop_size=512,
            round=False,
            fs=44100,
            output_path=npy.replace(data_root, query_root).replace(
                ".npy", f".{query_file}.npy"
            ),
        )
        for npy in npys
    ]

    if pmap:
        process_map(get_query_one, inout, chunksize=2, max_workers=24)
    else:
        for io in tqdm(inout):
            get_query_one(io)


def get_durations(data_root="/home/kwatchar3/Documents/data/moisesdb/npy"):
    npys = glob.glob(os.path.join(data_root, "**/mixture.npy"), recursive=True)

    durations = []

    for npy in tqdm(npys):
        audio = np.load(npy, mmap_mode="r")
        song_id = os.path.basename(os.path.dirname(npy))
        track_id = os.path.basename(npy).split(".")[0]

        durations.append(
            {
                "song_id": song_id,
                "track_id": track_id,
                "duration": audio.shape[-1] / 44100,
            }
        )

    durations = pd.DataFrame.from_records(durations)

    durations.to_csv(
        os.path.join(os.path.dirname(data_root), "durations.csv"), index=False
    )

    return durations


def clean_query_root(
    data_root="/home/kwatchar3/Documents/data/moisesdb/npy",
    query_root="/home/kwatchar3/Documents/data/moisesdb/npyq",
):
    npys = glob.glob(os.path.join(data_root, "**/*.query.npy"), recursive=True)

    for npy in tqdm(npys):
        dst = npy.replace(data_root, query_root)
        dstdir = os.path.dirname(dst)
        os.makedirs(dstdir, exist_ok=True)
        shutil.move(npy, dst)


def make_test_indices(
    metadata_path="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/metadata.csv",
    stem_path="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/stems.csv",
    splits_path="/storage/home/hcoda1/1/kwatchar3/data/data/moisesdb/splits.csv",
    test_split=5,
):
    
    coarse_stems = set(taxonomy.keys())
    fine_stems = set(chain(*taxonomy.values()))

    metadata = pd.read_csv(metadata_path)
    splits = pd.read_csv(splits_path)
    stems = pd.read_csv(stem_path)

    file_in_test = splits[splits["split"] == test_split]["song_id"].tolist()
    
    stems_test = stems[stems["song_id"].isin(file_in_test)]
    metadata_test = metadata[metadata["song_id"].isin(file_in_test)]
    splits_test = splits[splits["split"] == test_split]
    
    stems_test = stems_test.set_index("song_id")
    metadata_test = metadata_test.drop_duplicates("song_id").set_index("song_id")
    splits_test = splits_test.set_index("song_id")
    
    stem_to_song_id = defaultdict(list)
    song_id_to_stem = defaultdict(list)
    
    for song_id in file_in_test:
        
        stems_ = stems_test.loc[song_id]
        stem_names = stems_.T
        stem_names = stem_names[stem_names == 1].index.tolist()
        
        for stem in stem_names:
            stem_to_song_id[stem].append(song_id)
            
        song_id_to_stem[song_id] = stem_names
        
        
    indices = []
    no_query = []
    
    for song_id in file_in_test:
        
        genre = metadata_test.loc[song_id, "genre"]
        # print(genre)
        artist = metadata_test.loc[song_id, "artist"]
        # print(artist)
        
        stems_ = song_id_to_stem[song_id]
        
        for stem in stems_:
            possible_query = stem_to_song_id[stem]
            possible_query = [p for p in possible_query if p != song_id]
            
            if len(possible_query) == 0:
                print(f"No possible query for {song_id} with {stem}")
                
                no_query.append(
                    {
                        "song_id": song_id,
                        "stem": stem
                    }
                )
                continue
            
            query_df = metadata_test.loc[possible_query, ["genre", "artist"]]
            
            assert len(query_df) > 0
            
            query_df_ = query_df.copy()
            
            same_genre = True
            different_artist = True
            query_df = query_df[(query_df["genre"] == genre) & (query_df["artist"] != artist)]
            
            if len(query_df) == 0:
                
                same_genre = False
                different_artist = True
                
                query_df = query_df_.copy()
                query_df = query_df[(query_df["artist"] != artist)]
            
            if len(query_df) == 0:
                
                same_genre = True
                different_artist = False
                
                query_df = query_df_.copy()
                query_df = query_df[(query_df["genre"] == genre)]
            
            if len(query_df) == 0:
                
                same_genre = False
                different_artist = False
                
                query_df = query_df_.copy()
            
            query_id = query_df.sample(1).index[0]
            
            indices.append(
                {
                    "song_id": song_id,
                    "query_id": query_id,
                    "stem": stem,
                    "same_genre": same_genre,
                    "different_artist": different_artist
                }   
            )
            
    indices = pd.DataFrame.from_records(indices)
    no_query = pd.DataFrame.from_records(no_query)
    
    indices.to_csv(
        os.path.join(os.path.dirname(metadata_path), "test_indices.csv"), index=False
    )
    
    no_query.to_csv(
        os.path.join(os.path.dirname(metadata_path), "no_query.csv"), index=False
    )
    
    print("Total number of queries:", len(indices))
    print("Total number of no queries:", len(no_query))
    
    query_type = indices.groupby(["same_genre", "different_artist"]).size()
    
    print(query_type)


if __name__ == "__main__":
    import fire

    fire.Fire()
