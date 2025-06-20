from .generate_voices_bin import id2speaker


def generate_kokoro_v_11():
    s = f"""
# kokoro-multi-lang-v1_1

## Info about this model

This model is kokoro v1.1-zh and it is from <https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh>

It supports both `Chinese` and `English`.

| Number of speakers | Sample rate |
|--------------------|-------------|
| {len(id2speaker)} | 24000|

### Meaning of speaker prefix

|Prefix|Meaning| sid range| Number of speakers|
|---|---|---|---|
|af | American female|0 - 1| 2|
|bf| British female| 2| 1 |
|zf| Chinese female| 3 - 57 | 55 |
|zm| Chinese male| 58 - 102 | 45 |

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

    This model supports both Chinese and English. 小米的核心价值观是什么？答案
    是真诚热爱！有困难，请拨打110 或者18601200909。I am learning 机器学习.
    我在研究 machine learning。What do you think 中英文说的如何呢?
    今天是 2025年6月18号.

sample audios for different speakers are listed below:

"""
    for sid, name in id2speaker.items():
        s += f"\n### Speaker {sid} - {name}\n"
        s += f"""\n<audio controls>
  <source src="/sherpa/onnx/tts/all/kokoro/v1.1-zh/mp3//{sid}-{name}.mp3" type="audio/mp3">
</audio>\n\n"""

    with open("book/src/Chinese-English/kokoro-multi-lang-v1_1.md", "w") as f:
        f.write(s)
