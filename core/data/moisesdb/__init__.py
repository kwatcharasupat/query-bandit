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


def clean_track_inst(inst):

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


taxonomy = {k: [clean_track_inst(i.replace(" ", "_")) for i in v] for k, v in taxonomy.items()}