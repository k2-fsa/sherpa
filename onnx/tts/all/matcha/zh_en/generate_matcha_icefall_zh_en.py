from pathlib import Path
import jinja2


def get_data():
    model_dir = "matcha-icefall-zh-en"
    text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。在这次vocation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。在这个快速发展的时代，人工智能技术正在改变我们的生活方式。语音合成作为人工智能的重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"
    d = {
        "model_dir": model_dir,
        "acoustic_model": f"{model_dir}/model-steps-3.onnx",
        "data_dir": f"{model_dir}/espeak-ng-data",
        "tokens": f"{model_dir}/tokens.txt",
        "lexicon": f"{model_dir}/lexicon.txt",
        "vocoder": "vocos-16khz-univ.onnx",
        "text": text,
        "rule_fsts": f"{model_dir}/phone-zh.fst,{model_dir}/date-zh.fst,{model_dir}/number-zh.fst",
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
    d["name"] = "test_zh_en"

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
    d["name"] = "test-zh-en"
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
    d["name"] = "test-zh-en"
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
    d = {"url": "https://huggingface.co/spaces/k2-fsa/web-assembly-zh-en-tts-matcha"}

    t = read_file("./hf-space-wasm.md")
    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(**d)

    return template


def rust_api():
    d = get_data()

    t = read_file("./rust-api-example.rs.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Rust API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Rust API.

```rust
{code_template}
```

Please refer to the [Rust API documentation](https://k2-fsa.github.io/sherpa/onnx/rust-api/index.html)
for how to build and run the above Rust example.

</details>
    """

    return s


def android_apk(sherpa_onnx_version: str):
    v = sherpa_onnx_version
    d = get_data()
    d["apk"] = {
        "arm64-v8a": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-arm64-v8a-zh_en-tts-engine-matcha-icefall-zh-en.apk",
        "armeabi-v7a": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-armeabi-v7a-zh_en-tts-engine-matcha-icefall-zh-en.apk",
        "x86_64": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-x86_64-zh_en-tts-engine-matcha-icefall-zh-en.apk",
        "x86": f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}/sherpa-onnx-{v}-x86-zh_en-tts-engine-matcha-icefall-zh-en.apk",
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


def node_addon_api():
    d = get_data()

    t = read_file("./node-addon-api-example.js.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Node.js (addon) API

<details>
<summary>Click to expand</summary>

You need to install the `sherpa-onnx-node` npm package first:

```bash
npm install sherpa-onnx-node
```

You can use the following code to play with `{d['model_dir']}` with the Node.js addon API.

```javascript
{code_template}
```

Please refer to the [Node.js addon API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-addon-examples)
for more details.

</details>
    """

    return s


def dart_api():
    d = get_data()

    t = read_file("./dart-api-example.dart.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Dart API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Dart API.

```dart
{code_template}
```

Please refer to the [Dart API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples)
for more details.

</details>
    """

    return s


def swift_api():
    d = get_data()

    t = read_file("./swift-api-example.swift.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Swift API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Swift API.

```swift
{code_template}
```

Please refer to the [Swift API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/swift-api-examples)
for more details.

</details>
    """

    return s


def csharp_api():
    d = get_data()

    t = read_file("./csharp-api-example.cs.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## C# API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with C# API.

```c#
{code_template}
```

Please refer to the [C# API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet)
for more details.

</details>
    """

    return s


def kotlin_api():
    d = get_data()

    t = read_file("./kotlin-api-example.kt.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Kotlin API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Kotlin API.

```kotlin
{code_template}
```

Please refer to the [Kotlin API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/kotlin-api-examples)
for more details.

</details>
    """

    return s


def java_api():
    d = get_data()

    t = read_file("./java-api-example.java.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Java API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Java API.

```java
{code_template}
```

Please refer to the [Java API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples)
for more details.

</details>
    """

    return s


def pascal_api():
    d = get_data()

    t = read_file("./pascal-api-example.pas.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Pascal API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Pascal API.

```pascal
{code_template}
```

Please refer to the [Pascal API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples)
for more details.

</details>
    """

    return s


def go_api():
    d = get_data()

    t = read_file("./go-api-example.go.in")

    environment = jinja2.Environment()
    code_template = environment.from_string(t)

    code_template = code_template.render(**d)

    s = f"""
## Go API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{d['model_dir']}` with Go API.

```go
{code_template}
```

Please refer to the [Go API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples)
for more details.

</details>
    """

    return s


def generate_icefall_zh_en_matcha(sherpa_onnx_version: str):
    d = get_data()
    s = f"""
# matcha-icefall-zh-en

||||||
|---|---|---|---|---|
|[Info about this model](#info-about-this-model)|[Download the model](#download-the-model)|[HF Space](#hf-space)|[Android APK](#android-apk)|[Python API](#python-api)|
|[C API](#c-api)|[C++ API](#c-api-1)|[Rust API](#rust-api)|[Node.js API](#nodejs-addon-api)|[Dart API](#dart-api)|
|[Swift API](#swift-api)|[C# API](#c-api-2)|[Kotlin API](#kotlin-api)|[Java API](#java-api)|[Pascal API](#pascal-api)|
|[Go API](#go-api)|[Samples](#samples)||||

{read_file('./info.md')}
"""

    s += read_file("./download.md")
    s += "\n"
    s += hf_space()
    s += "\n"
    s += hf_space_wasm()
    s += android_apk(sherpa_onnx_version)
    s += python_api()
    s += c_api()
    s += cxx_api()
    s += rust_api()
    s += node_addon_api()
    s += dart_api()
    s += swift_api()
    s += csharp_api()
    s += kotlin_api()
    s += java_api()
    s += pascal_api()
    s += go_api()

    s += f"""

## Samples

For the following text:

```
{d['text']}
```

sample audios for different speakers are listed below:

"""
    s += f"\n### Speaker 0\n"
    s += f"""\n<audio controls>
<source src="/sherpa/onnx/tts/all/matcha/icefall-zh-en/mp3/0.mp3" type="audio/mp3">
</audio>\n\n"""

    with open("book/src/Chinese-English/matcha-icefall-zh-en.md", "w") as f:
        f.write(s)
