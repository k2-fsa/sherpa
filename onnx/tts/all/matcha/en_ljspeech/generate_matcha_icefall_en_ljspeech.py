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


def generate_icefall_en_ljspeech_matcha(sherpa_onnx_version: str):
    s = f"""
# matcha-icefall-en_US-ljspeech

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

    s += """

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
