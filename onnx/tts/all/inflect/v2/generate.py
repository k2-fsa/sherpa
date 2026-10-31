import jinja2

# Maps variant to (model_dir, model_name, mp3_version_name)
_V2_MODELS = {
    "nano": ("vits-inflect-en-nano-v2", "model.onnx", "vits-inflect-en-nano-v2"),
    "micro": ("vits-inflect-en-micro-v2", "model.onnx", "vits-inflect-en-micro-v2"),
}


def get_android_apk(variant, sherpa_onnx_version):
    model_dir = _V2_MODELS[variant][0]

    v = sherpa_onnx_version
    url = f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}"
    url_cn = f"https://hf-mirror.com/csukuangfj2/sherpa-onnx-apk/blob/main/tts-engine-new/{v}"

    apk = dict()
    apk_cn = dict()
    for arch in ["arm64-v8a", "armeabi-v7a", "x86_64", "x86"]:
        apk[arch] = (
            f"{url}/sherpa-onnx-{v}-{arch}-eng-tts-engine-{model_dir}.apk"
        )
        apk_cn[arch] = (
            f"{url_cn}/sherpa-onnx-{v}-{arch}-eng-tts-engine-{model_dir}.apk"
        )

    s = f"""

## Android APK

<details>
<summary>Click to expand</summary>

The following table shows the Android TTS Engine APK with this model
for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) v{v}

| ABI | URL | 中国镜像|
|---|---|---|
|arm64-v8a|[Download]({apk['arm64-v8a']})|[下载]({apk_cn['arm64-v8a']})|
|armeabi-v7a|[Download]({apk['armeabi-v7a']})|[下载]({apk_cn['armeabi-v7a']})|
|x86_64|[Download]({apk['x86_64']})|[下载]({apk_cn['x86_64']})|
|x86|[Download]({apk['x86']})|[下载]({apk_cn['x86']})|

> If you don't know what ABI is, you probably need to select `arm64-v8a`.

The source code for the APK can be found at

<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxTtsEngine>

Please refer to the [documentation](https://k2-fsa.github.io/sherpa/onnx/android/index.html)
for how to build the APK from source code.

More Android APKs can be found at

<https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html>

</details>
    """
    return s


def _get_model_info(variant):
    model_dir, model_name, _ = _V2_MODELS[variant]
    d = {
        "model": f"./{model_dir}/{model_name}",
        "data_dir": f"./{model_dir}/espeak-ng-data",
        "tokens": f"./{model_dir}/tokens.txt",
        "text": "Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.",
    }
    return model_dir, model_name, d


def get_c_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/c-api-example.c.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## C API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with C API.

```c
{template}
```

In the following, we describe how to compile and run the above C example.

### Use shared library (dynamic link)

```bash
cd /tmp
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build-shared
cd build-shared

cmake \\
 -DSHERPA_ONNX_ENABLE_C_API=ON \\
 -DCMAKE_BUILD_TYPE=Release \\
 -DBUILD_SHARED_LIBS=ON \\
 -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-onnx/shared \\
 ..

make
make install
```

You can find required header file and library files inside ``/tmp/sherpa-onnx/shared``.

Assume you have saved the above example file as `/tmp/test-inflect.c`.
Then you can compile it with the following command:

```bash
gcc \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-inflect \\
  /tmp/test-inflect.c
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-inflect
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-inflect`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """

    return s


def get_python_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/python-api.py.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Python API

<details>
<summary>Click to expand</summary>

Assume you have installed `sherpa-onnx` via
```bash
pip install sherpa-onnx
```
and you have downloaded the model from

<https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{model_dir}.tar.bz2>

You can use the following code to play with `{model_dir}`

```python
{template}
```

</details>
    """

    return s


def get_cxx_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/cxx-api-example.cc.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## C++ API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with C++ API.

```c++
{template}
```

In the following, we describe how to compile and run the above C++ example.

### Use shared library (dynamic link)

```bash
cd /tmp
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build-shared
cd build-shared

cmake \\
 -DSHERPA_ONNX_ENABLE_C_API=ON \\
 -DCMAKE_BUILD_TYPE=Release \\
 -DBUILD_SHARED_LIBS=ON \\
 -DCMAKE_INSTALL_PREFIX=/tmp/sherpa-onnx/shared \\
 ..

make
make install
```

You can find required header file and library files inside ``/tmp/sherpa-onnx/shared``.

Assume you have saved the above example file as `/tmp/test-inflect.cc`.
Then you can compile it with the following command:

```bash
g++ \\
  -std=c++17 \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-inflect \\
  /tmp/test-inflect.cc
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-inflect
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-inflect`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """

    return s


def get_rust_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/rust-api-example.rs.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Rust API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Rust API.

```rust
{template}
```

Please refer to the [Rust API documentation](https://k2-fsa.github.io/sherpa/onnx/rust-api/index.html)
for how to build and run the above Rust example.

</details>
    """

    return s


def get_node_addon_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/node-addon-api-example.js.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Node.js (addon) API

<details>
<summary>Click to expand</summary>

You need to install the `sherpa-onnx-node` npm package first:

```bash
npm install sherpa-onnx-node
```

You can use the following code to play with `{model_dir}` with the Node.js addon API.

```javascript
{template}
```

Please refer to the [Node.js addon API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-addon-examples)
for more details.

</details>
    """

    return s


def get_dart_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/dart-api-example.dart.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Dart API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Dart API.

```dart
{template}
```

Please refer to the [Dart API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples)
for more details.

</details>
    """

    return s


def get_swift_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/swift-api-example.swift.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Swift API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Swift API.

```swift
{template}
```

Please refer to the [Swift API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/swift-api-examples)
for more details.

</details>
    """

    return s


def get_csharp_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/csharp-api-example.cs.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## C# API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with C# API.

```c#
{template}
```

Please refer to the [C# API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet)
for more details.

</details>
    """

    return s


def get_kotlin_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/kotlin-api-example.kt.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Kotlin API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Kotlin API.

```kotlin
{template}
```

Please refer to the [Kotlin API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/kotlin-api-examples)
for more details.

</details>
    """

    return s


def get_java_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/java-api-example.java.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Java API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Java API.

```java
{template}
```

Please refer to the [Java API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples)
for more details.

</details>
    """

    return s


def get_pascal_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/pascal-api-example.pas.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Pascal API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Pascal API.

```pascal
{template}
```

Please refer to the [Pascal API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples)
for more details.

</details>
    """

    return s


def get_go_api(variant):
    model_dir, model_name, d = _get_model_info(variant)

    with open("./inflect/templates/go-api-example.go.in") as f:
        t = f.read()

    environment = jinja2.Environment()
    template = environment.from_string(t)

    template = template.render(
        **d,
        sid=0,
    )

    s = f"""
## Go API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `{model_dir}` with Go API.

```go
{template}
```

Please refer to the [Go API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples)
for more details.

</details>
    """

    return s


def generate_inflect_v2(variant: str, sherpa_onnx_version: str = "1.13.4"):
    """Generate doc page for an inflect v2 model variant.

    Args:
        variant: One of "nano", "micro"
        sherpa_onnx_version: e.g., "1.13.4"
    """
    model_dir, model_name, mp3_version_name = _V2_MODELS[variant]

    hf_repo = f"Inflect-{'Nano' if variant == 'nano' else 'Micro'}-v2"
    hf_url = f"https://huggingface.co/owensong/{hf_repo}"
    download_url = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{model_dir}.tar.bz2"

    s = f"""
# {model_dir}

||||||
|---|---|---|---|---|
|[Info about this model](#info-about-this-model)|[Download the model](#download-the-model)|[Android APK](#android-apk)|[Python API](#python-api)|[C API](#c-api)|
|[C++ API](#c-api-1)|[Rust API](#rust-api)|[Node.js API](#nodejs-addon-api)|[Dart API](#dart-api)|[Swift API](#swift-api)|
|[C# API](#c-api-2)|[Kotlin API](#kotlin-api)|[Java API](#java-api)|[Pascal API](#pascal-api)|[Go API](#go-api)|
|[Samples](#samples)|||||

## Info about this model

This model is from <{hf_url}>

It supports only `English`.

| Number of speakers | Sample rate |
|--------------------|-------------|
| 1 | 24000|

## Download the model

<details>
<summary>Click to expand</summary>

Model download address

<{download_url}>

</details>

"""
    s += get_android_apk(variant, sherpa_onnx_version)

    s += get_python_api(variant)
    s += get_c_api(variant)
    s += get_cxx_api(variant)
    s += get_rust_api(variant)
    s += get_node_addon_api(variant)
    s += get_dart_api(variant)
    s += get_swift_api(variant)
    s += get_csharp_api(variant)
    s += get_kotlin_api(variant)
    s += get_java_api(variant)
    s += get_pascal_api(variant)
    s += get_go_api(variant)

    s += """
## Samples

For the following text:

    Friends fell out often because life was changing so fast.
    The easiest thing in the world was to lose touch with someone.

sample audio:

<audio controls>
    <source src="/sherpa/onnx/tts/all/inflect/""" + mp3_version_name + """/mp3/0.mp3" type="audio/mp3">
</audio>

"""

    with open(f"book/src/English/{model_dir}.md", "w") as f:
        f.write(s)
