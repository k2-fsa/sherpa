def generate_icefall_zh_matcha():
    s = f"""
# matcha-icefall-zh-baker

## Info about this model

This model is trained using the code from <https://github.com/k2-fsa/icefall/tree/master/egs/baker_zh/TTS/matcha>

It supports only `Chinese`.

| Number of speakers | Sample rate |
|--------------------|-------------|
| 1 | 22050|

## Samples

For the following text:

    某某银行的副行长和一些行政领导表示，他们去过长江和长白山;
    经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。
    当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，
    思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，
    沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔.

sample audios for different speakers are listed below:

"""
    s += f"\n### Speaker 0\n"
    s += f"""\n<audio controls>
<source src="/sherpa/onnx/tts/all/matcha/icefall-zh/mp3/0.mp3" type="audio/mp3">
</audio>\n\n"""

    with open("book/src/Chinese/matcha-icefall-zh-baker.md", "w") as f:
        f.write(s)
