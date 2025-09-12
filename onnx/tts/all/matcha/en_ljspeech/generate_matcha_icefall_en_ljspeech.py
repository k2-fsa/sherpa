from pathlib import Path


def generate_icefall_en_ljspeech_matcha():
    s = f"""
# matcha-icefall-en_US-ljspeech

## Info about this model

This model is trained using the code from <https://github.com/k2-fsa/icefall/tree/master/egs/ljspeech/TTS/matcha>

It supports only `English`.

| Number of speakers | Sample rate |
|--------------------|-------------|
| 1 | 22050|

## Samples

For the following text:

    Friends fell out often because life was changing so fast.
    The easiest thing in the world was to lose touch with someone.

sample audios for different speakers are listed below:

"""
    s += f"\n### Speaker 0\n"
    s += f"""\n<audio controls>
<source src="/sherpa/onnx/tts/all/matcha/icefall-en-ljspeech/mp3/0.mp3" type="audio/mp3">
</audio>\n\n"""

    Path(f"./book/src/English").mkdir(parents=True, exist_ok=True)
    with open("book/src/English/matcha-icefall-en_US-ljspeech.md", "w") as f:
        f.write(s)
