from pathlib import Path

import jinja2


def get_data():
    model_dir = "matcha-icefall-en_US-ljspeech"
    text = "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone."

    d = {
        "model_dir": model_dir,
        "acoustic_model": f"{model_dir}/model-steps-3.onnx",
        "data_dir": f"{model_dir}/espeak-ng-data",
        "tokens": f"{model_dir}/tokens.txt",
        "vocoder": "vocos-22khz-univ.onnx",
        "text": text,
    }

    return d


def read_file(name):
    pwd = Path(__file__).parent.resolve()
    with open(f"{pwd}/{name}") as f:
        s = f.read()

    return s


def python_api():
    d = get_data()

    t = read_file("./generate_samples.py.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    d["code"] = code_template
    d["name"] = "test-en"

    t = read_file("./python-api.md")
    template = environment.from_string(t)

    template = template.render(**d)

    s = f"""
{template}
    """

    return s


def c_api():
    d = get_data()

    t = read_file("./c-api-example.c.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    d["code"] = code_template

    d["build"] = read_file("./build.md")
    d["name"] = "test-en"
    d["env"] = read_file("./set-env.md").strip()

    t = read_file("./c-api.md")
    template = environment.from_string(t)

    template = template.render(**d)

    s = f"""
{template}
    """

    return s


def cxx_api():
    d = get_data()

    t = read_file("./cxx-api-example.cc.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    d["code"] = code_template

    d["build"] = read_file("./build.md")
    d["name"] = "test-en"
    d["env"] = read_file("./set-env.md").strip()

    t = read_file("./cxx-api.md")
    template = environment.from_string(t)

    template = template.render(**d)

    s = f"""
{template}
    """

    return s


def hf_space():
    s = read_file("./hf-space.md")
    return s


def hf_space_wasm():
    d = {"url": "https://huggingface.co/spaces/k2-fsa/web-assembly-en-tts-matcha"}

    t = read_file("./hf-space-wasm.md")
    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(**d)

    return template


def android_apk(sherpa_onnx_version: str):
    v = sherpa_onnx_version
    d = get_data()
    d["apk"] = {
        "arm64-v8a": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-arm64-v8a-en-tts-engine-matcha-icefall-en_US-ljspeech.apk",
        "armeabi-v7a": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-armeabi-v7a-en-tts-engine-matcha-icefall-en_US-ljspeech.apk",
        "x86_64": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-x86_64-en-tts-engine-matcha-icefall-en_US-ljspeech.apk",
        "x86": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-x86-en-tts-engine-matcha-icefall-en_US-ljspeech.apk",
    }

    d["apk_cn"] = {}
    for k, v in d["apk"].items():
        d["apk_cn"][k] = v.replace("huggingface.co", "hf-mirror.com").replace(
            "resolve", "blob"
        )

    t = read_file("./android.md")

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(**d)

    s = f"""
{template}
    """

    return s


def generate_icefall_en_ljspeech_matcha(sherpa_onnx_version: str):
    s = f"""
# matcha-icefall-en_US-ljspeech

{read_file('./info.md')}

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

    s += android_apk(sherpa_onnx_version)
    s += "\n"
    s += hf_space()
    s += "\n"
    s += hf_space_wasm()
    s += read_file("./download.md")
    s += python_api()
    s += c_api()
    s += cxx_api()

    Path(f"./book/src/English").mkdir(parents=True, exist_ok=True)
    with open("book/src/English/matcha-icefall-en_US-ljspeech.md", "w") as f:
        f.write(s)
