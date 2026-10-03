from pathlib import Path

import jinja2


def _read_file(name):
    pwd = Path(__file__).parent.resolve()
    with open(f"{pwd}/../templates/{name}") as f:
        return f.read()


def _get_data():
    model_dir = "nabra-82m-arabic-int8"
    text = "تحول تقنية تحويل النص إلى كلام الجمل إلى صوت واضحا."
    return {
        "model": f"{model_dir}/model.int8.onnx",
        "voices": f"{model_dir}/voices.bin",
        "tokens": f"{model_dir}/tokens.txt",
        "data_dir": f"{model_dir}/espeak-ng-data",
        "text": text,
        "lang": "ar",
    }


def _android_apk():
    import os

    v = os.environ.get("SHERPA_ONNX_VERSION", "1.13.7")
    url = f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}"
    url_cn = f"https://hf-mirror.com/csukuangfj2/sherpa-onnx-apk/blob/main/tts-engine-new/{v}"
    model = "nabra-82m-arabic-int8"
    lang = "ara"

    apk = dict()
    apk_cn = dict()
    for arch in ["arm64-v8a", "armeabi-v7a", "x86_64", "x86"]:
        apk[arch] = f"{url}/sherpa-onnx-{v}-{arch}-{lang}-tts-engine-{model}.apk"
        apk_cn[arch] = f"{url_cn}/sherpa-onnx-{v}-{arch}-{lang}-tts-engine-{model}.apk"

    return f"""
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


def _download():
    return """
## Download the model

<details>
<summary>Click to expand</summary>

Model download address

<https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/nabra-82m-arabic-int8.tar.bz2>

</details>
    """


def _c_api():
    d = _get_data()
    t = _read_file("c-api-example.c.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with C API.

```c
{template}
```

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

Assume you have saved the above example file as `/tmp/test-nabra.c`.
Then you can compile it with the following command:

```bash
gcc \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-nabra \\
  /tmp/test-nabra.c
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-nabra
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-nabra`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """


def _cxx_api():
    d = _get_data()
    t = _read_file("cxx-api-example.cc.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C++ API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with C++ API.

```c++
{template}
```

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

Assume you have saved the above example file as `/tmp/test-nabra.cc`.
Then you can compile it with the following command:

```bash
g++ \\
  -std=c++17 \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-cxx-api \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-nabra \\
  /tmp/test-nabra.cc
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-nabra
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-nabra`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """


def _python_api():
    d = _get_data()
    t = _read_file("python-api.py.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Python API

<details>
<summary>Click to expand</summary>

Assume you have installed `sherpa-onnx` via
```bash
pip install sherpa-onnx
```
and you have downloaded the model from

<https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/nabra-82m-arabic-int8.tar.bz2>

You can use the following code to play with `nabra-82m-arabic-int8`

```python
{template}
```

</details>
    """


def _rust_api():
    d = _get_data()
    t = _read_file("rust-api-example.rs.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Rust API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Rust API.

```rust
{template}
```

Please refer to the [Rust API documentation](https://k2-fsa.github.io/sherpa/onnx/rust-api/index.html)
for how to build and run the above Rust example.

</details>
    """


def _node_addon_api():
    d = _get_data()
    t = _read_file("node-addon-api-example.js.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Node.js (addon) API

<details>
<summary>Click to expand</summary>

You need to install the `sherpa-onnx-node` npm package first:

```bash
npm install sherpa-onnx-node
```

You can use the following code to play with `nabra-82m-arabic-int8` with the Node.js addon API.

```javascript
{template}
```

Please refer to the [Node.js addon API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-addon-examples)
for more details.

</details>
    """


def _dart_api():
    d = _get_data()
    t = _read_file("dart-api-example.dart.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Dart API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Dart API.

```dart
{template}
```

Please refer to the [Dart API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples)
for more details.

</details>
    """


def _swift_api():
    d = _get_data()
    t = _read_file("swift-api-example.swift.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Swift API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Swift API.

```swift
{template}
```

Please refer to the [Swift API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/swift-api-examples)
for more details.

</details>
    """


def _csharp_api():
    d = _get_data()
    t = _read_file("csharp-api-example.cs.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C# API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with C# API.

```c#
{template}
```

Please refer to the [C# API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet)
for more details.

</details>
    """


def _kotlin_api():
    d = _get_data()
    t = _read_file("kotlin-api-example.kt.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Kotlin API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Kotlin API.

```kotlin
{template}
```

Please refer to the [Kotlin API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/kotlin-api-examples)
for more details.

</details>
    """


def _java_api():
    d = _get_data()
    t = _read_file("java-api-example.java.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Java API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Java API.

```java
{template}
```

Please refer to the [Java API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples)
for more details.

</details>
    """


def _pascal_api():
    d = _get_data()
    t = _read_file("pascal-api-example.pas.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Pascal API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Pascal API.

```pascal
{template}
```

Please refer to the [Pascal API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples)
for more details.

</details>
    """


def _go_api():
    d = _get_data()
    t = _read_file("go-api-example.go.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d, sid=0)

    return f"""
## Go API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `nabra-82m-arabic-int8` with Go API.

```go
{template}
```

Please refer to the [Go API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples)
for more details.

</details>
    """


def generate_nabra():
    s = f"""
# kokoro-nabra-82m-arabic-int8

||||||
|---|---|---|---|---|
|[Info about this model](#info-about-this-model)|[Download the model](#download-the-model)|[Android APK](#android-apk)|[Python API](#python-api)|[C API](#c-api)|
|[C++ API](#c-api-1)|[Rust API](#rust-api)|[Node.js API](#nodejs-addon-api)|[Dart API](#dart-api)|[Swift API](#swift-api)|
|[C# API](#c-api-2)|[Kotlin API](#kotlin-api)|[Java API](#java-api)|[Pascal API](#pascal-api)|[Go API](#go-api)|
|[Samples](#samples)|||||

## Info about this model

This model is nabra 82M from <https://huggingface.co/marwanelamami/nabra-82m-sherpa-onnx>

It supports only **Arabic**.

| Number of speakers | Sample rate |
|--------------------|-------------|
| 1 | 24000|

"""

    s += _download()
    s += _android_apk()
    s += _python_api()
    s += _c_api()
    s += _cxx_api()
    s += _rust_api()
    s += _node_addon_api()
    s += _dart_api()
    s += _swift_api()
    s += _csharp_api()
    s += _kotlin_api()
    s += _java_api()
    s += _pascal_api()
    s += _go_api()

    s += """
## Samples

For the following text:

    تحول تقنية تحويل النص إلى كلام الجمل إلى صوت واضحا.

sample audio:

<audio controls>
  <source src="/sherpa/onnx/tts/all/kokoro/nabra/mp3/0.mp3" type="audio/mp3">
</audio>

"""

    Path("./book/src/Arabic").mkdir(parents=True, exist_ok=True)
    with open("book/src/Arabic/kokoro-nabra-82m-arabic-int8.md", "w") as f:
        f.write(s)
