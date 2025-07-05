from .generate_voices_bin import id2speaker


def generate_kokoro_v019():
    s = f"""
# kokoro-en-v0_19

## Info about this model

This model is kokoro v0.19 and it is from <https://huggingface.co/hexgrad/kLegacy>

It supports only `English`.

| Number of speakers | Sample rate |
|--------------------|-------------|
| {len(id2speaker)} | 24000|

### Meaning of speaker prefix

|Prefix|Meaning| sid range| Number of speakers|
|---|---|---|---|
|af | American female|0 - 4| 5|
|am| American male| 5 - 6| 2|
|bf| British female| 7 - 8 | 2 |
|bm| British male| 9 - 10 | 2|

### speaker ID to speaker name (sid -> name)
The mapping from speaker ID (sid) to speaker name is given below:

"""

    num_per_line = 4

    t = "|" * (num_per_line + 2)
    t += "\n"
    t += "|---" * (num_per_line + 1)
    t += "|\n"

    t += f"|0 - {num_per_line-1}"

    for i, (sid, name) in enumerate(id2speaker.items()):
        t += f"|{sid} -> {name}"
        if (i + 1) % num_per_line == 0:
            t += "|\n"
            if (i + 1) == min(i + num_per_line, len(id2speaker) - 1):
                t += f"|{i+1}"
            else:
                t += f"|{i+1} - {min(i+num_per_line, len(id2speaker)-1)}"
    while (i + 1) % num_per_line != 0:
        t += "|"
        i += 1
    t += "|\n"

    s += t

    s += """
### speaker name to speaker ID (name -> sid)
The mapping from speaker name to speaker ID (sid) is given below:

"""

    t = "|" * (num_per_line + 2)
    t += "\n"
    t += "|---" * (num_per_line + 1)
    t += "|\n"

    t += f"|0 - {num_per_line-1}"

    for i, (sid, name) in enumerate(id2speaker.items()):
        t += f"|{name} -> {sid}"
        if (i + 1) % num_per_line == 0:
            t += "|\n"
            t += f"|{i+1} - {min(i+num_per_line, len(id2speaker)-1)}"
    while (i + 1) % num_per_line != 0:
        t += "|"
        i += 1
    t += "|\n"

    s += t

    s += """
## Samples

For the following text:

    Friends fell out often because life was changing so fast.
    The easiest thing in the world was to lose touch with someone.

sample audios for different speakers are listed below:

"""
    for sid, name in id2speaker.items():
        s += f"\n### Speaker {sid} - {name}\n"
        s += f"""\n<audio controls>
  <source src="/sherpa/onnx/tts/all/kokoro/v0.19/mp3//{sid}-{name}.mp3" type="audio/mp3">
</audio>\n\n"""

    with open("book/src/English/kokoro-en-v0_19.md", "w") as f:
        f.write(s)
