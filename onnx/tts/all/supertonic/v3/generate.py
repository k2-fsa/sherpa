from pathlib import Path

import jinja2


def _read_file(name):
    pwd = Path(__file__).parent.resolve()
    with open(f"{pwd}/../templates/{name}") as f:
        return f.read()


# lang code -> (language name, sample text, number of sentences)
_LANG_INFO = {
    "en": ("English", "How are you doing today? This is a text-to-speech engine using next generation Kaldi.", 20),
    "ko": ("Korean", "이것은 차세대 kaldi를 사용하는 텍스트 음성 변환 엔진입니다", 20),
    "ja": ("Japanese", "これは次世代のkaldiを使用したテキスト読み上げエンジンです", 10),
    "ar": ("Arabic", "هذا هو محرك تحويل النص إلى كلام باستخدام الجيل القادم من كالدي", 10),
    "bg": ("Bulgarian", "Това е машина за преобразуване на текст в реч, използваща Kaldi от следващо поколение", 10),
    "cs": ("Czech", "Toto je převodník textu na řeč využívající novou generaci kaldi", 10),
    "da": ("Danish", "Dette er en tekst til tale-motor, der bruger næste generation af kaldi", 10),
    "de": ("German", "Dies ist eine Text-to-Speech-Engine, die Kaldi der nächsten Generation verwendet", 10),
    "el": ("Greek", "Αυτή είναι μια μηχανή κειμένου σε ομιλία που χρησιμοποιεί kaldi επόμενης γενιάς", 10),
    "es": ("Spanish", "Este es un motor de texto a voz que utiliza kaldi de próxima generación.", 20),
    "et": ("Estonian", "See on teksti kõneks muutmise mootor, mis kasutab järgmise põlvkonna Kaldi", 10),
    "fi": ("Finnish", "Tämä on tekstistä puheeksi -moottori, joka käyttää seuraavan sukupolven kaldia", 10),
    "fr": ("French", "Il s'agit d'un moteur de synthèse vocale utilisant Kaldi de nouvelle génération", 20),
    "hi": ("Hindi", "यह अगली पीढ़ी के काल्डी का उपयोग करने वाला एक टेक्स्ट-टू-स्पीच इंजन है", 10),
    "hr": ("Croatian", "Ovo je mehanizam za pretvaranje teksta u govor koji koristi Kaldi sljedeće generacije", 10),
    "hu": ("Hungarian", "Ez egy szövegfelolvasó motor a következő generációs kaldi használatával", 10),
    "id": ("Indonesian", "Ini adalah mesin text-to-speech yang menggunakan Kaldi generasi berikutnya", 10),
    "it": ("Italian", "Questo è un motore di sintesi vocale che utilizza kaldi di nuova generazione", 10),
    "lt": ("Lithuanian", "Tai teksto į kalbą variklis, kuriame naudojamas naujos kartos Kaldi", 10),
    "lv": ("Latvian", "Šis ir teksta pārvēršanas runā dzinējs, kas izmanto nākamās paaudzes Kaldi", 10),
    "nl": ("Dutch", "Dit is een tekst-naar-spraak-engine die gebruik maakt van Kaldi van de volgende generatie", 10),
    "pl": ("Polish", "Jest to silnik syntezatora mowy wykorzystujący Kaldi nowej generacji", 10),
    "pt": ("Portuguese", "Este é um mecanismo de conversão de texto em fala usando Kaldi de próxima geração", 20),
    "ro": ("Romanian", "Acesta este un motor text to speech care folosește generația următoare de kadi", 10),
    "ru": ("Russian", "Это движок преобразования текста в речь, использующий Kaldi следующего поколения.", 10),
    "sk": ("Slovak", "Toto je nástroj na prevod textu na reč využívajúci kaldi novej generácie", 10),
    "sl": ("Slovenian", "To je mehanizem za pretvorbo besedila v govor, ki uporablja Kaldi naslednje generacije", 10),
    "sv": ("Swedish", "Detta är en text till tal-motor som använder nästa generations kaldi", 10),
    "tr": ("Turkish", "Bu, yeni nesil kaldi'yi kullanan bir metinden konuşmaya motorudur", 10),
    "uk": ("Ukrainian", "Це механізм перетворення тексту на мовлення, який використовує kaldi нового покоління", 10),
    "vi": ("Vietnamese", "Đây là công cụ chuyển văn bản thành giọng nói sử dụng kaldi thế hệ tiếp theo", 10),
}


def _get_data(lang):
    name, text, _ = _LANG_INFO[lang]
    model_dir = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
    return {
        "model_dir": model_dir,
        "text": text,
        "lang": lang,
    }


def _get_sherpa_onnx_version():
    import os
    return os.environ.get("SHERPA_ONNX_VERSION", "1.13.1")


def _android_apk(lang):
    v = _get_sherpa_onnx_version()
    lang_iso_639_3 = _lang_to_iso639_3(lang)
    model = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"

    url = f"https://huggingface.co/csukuangfj2/sherpa-onnx-apk/resolve/main/tts-engine-new/{v}"
    url_cn = f"https://hf-mirror.com/csukuangfj2/sherpa-onnx-apk/blob/main/tts-engine-new/{v}"

    apk = {}
    apk_cn = {}
    for arch in ["arm64-v8a", "armeabi-v7a", "x86_64", "x86"]:
        apk[arch] = f"{url}/sherpa-onnx-{v}-{arch}-{lang_iso_639_3}-tts-engine-{model}.apk"
        apk_cn[arch] = f"{url_cn}/sherpa-onnx-{v}-{arch}-{lang_iso_639_3}-tts-engine-{model}.apk"

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


def _lang_to_iso639_3(lang):
    mapping = {
        "en": "eng", "ko": "kor", "ja": "jpn", "ar": "ara", "bg": "bul",
        "cs": "ces", "da": "dan", "de": "deu", "el": "ell", "es": "spa",
        "et": "est", "fi": "fin", "fr": "fra", "hi": "hin", "hr": "hrv",
        "hu": "hun", "id": "ind", "it": "ita", "lt": "lit", "lv": "lav",
        "nl": "nld", "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
        "sk": "slk", "sl": "slv", "sv": "swe", "tr": "tur", "uk": "ukr",
        "vi": "vie",
    }
    return mapping.get(lang, lang)


def _download():
    return """
## Download the model

<details>
<summary>Click to expand</summary>

Model download address

<https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2>

</details>
    """


def _c_api(lang):
    d = _get_data(lang)
    t = _read_file("c-api-example.c.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with C API.

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

Assume you have saved the above example file as `/tmp/test-supertonic.c`.
Then you can compile it with the following command:

```bash
gcc \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-supertonic \\
  /tmp/test-supertonic.c
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-supertonic
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-supertonic`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """


def _cxx_api(lang):
    d = _get_data(lang)
    t = _read_file("cxx-api-example.cc.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C++ API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with C++ API.

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

Assume you have saved the above example file as `/tmp/test-supertonic.cc`.
Then you can compile it with the following command:

```bash
g++ \\
  -std=c++17 \\
  -I /tmp/sherpa-onnx/shared/include \\
  -L /tmp/sherpa-onnx/shared/lib \\
  -lsherpa-onnx-cxx-api \\
  -lsherpa-onnx-c-api \\
  -lonnxruntime \\
  -o /tmp/test-supertonic \\
  /tmp/test-supertonic.cc
```

Now you can run
```bash
cd /tmp

# Assume you have downloaded the model and extracted it to /tmp
./test-supertonic
```

> You probably need to run
>    ```bash
>    # For Linux
>    export LD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$LD_LIBRARY_PATH
>
>    # For macOS
>    export DYLD_LIBRARY_PATH=/tmp/sherpa-onnx/shared/lib:$DYLD_LIBRARY_PATH
>    ```
>  before you run `/tmp/test-supertonic`.

### Use static library (static link)

Please see the documentation at

<https://k2-fsa.github.io/sherpa/onnx/c-api/index.html>

</details>
    """


def _python_api(lang):
    d = _get_data(lang)
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

<https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2>

You can use the following code to play with `supertonic-3`

```python
{template}
```

</details>
    """


def _rust_api(lang):
    d = _get_data(lang)
    t = _read_file("rust-api-example.rs.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Rust API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Rust API.

```rust
{template}
```

Please refer to the [Rust API documentation](https://k2-fsa.github.io/sherpa/onnx/rust-api/index.html)
for how to build and run the above Rust example.

</details>
    """


def _node_addon_api(lang):
    d = _get_data(lang)
    t = _read_file("node-addon-api-example.js.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Node.js (addon) API

<details>
<summary>Click to expand</summary>

You need to install the `sherpa-onnx-node` npm package first:

```bash
npm install sherpa-onnx-node
```

You can use the following code to play with `supertonic-3` with the Node.js addon API.

```javascript
{template}
```

Please refer to the [Node.js addon API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/nodejs-addon-examples)
for more details.

</details>
    """


def _dart_api(lang):
    d = _get_data(lang)
    t = _read_file("dart-api-example.dart.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Dart API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Dart API.

```dart
{template}
```

Please refer to the [Dart API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples)
for more details.

</details>
    """


def _swift_api(lang):
    d = _get_data(lang)
    t = _read_file("swift-api-example.swift.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Swift API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Swift API.

```swift
{template}
```

Please refer to the [Swift API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/swift-api-examples)
for more details.

</details>
    """


def _csharp_api(lang):
    d = _get_data(lang)
    t = _read_file("csharp-api-example.cs.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## C# API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with C# API.

```c#
{template}
```

Please refer to the [C# API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/dotnet)
for more details.

</details>
    """


def _kotlin_api(lang):
    d = _get_data(lang)
    t = _read_file("kotlin-api-example.kt.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Kotlin API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Kotlin API.

```kotlin
{template}
```

Please refer to the [Kotlin API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/kotlin-api-examples)
for more details.

</details>
    """


def _java_api(lang):
    d = _get_data(lang)
    t = _read_file("java-api-example.java.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Java API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Java API.

```java
{template}
```

Please refer to the [Java API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples)
for more details.

</details>
    """


def _pascal_api(lang):
    d = _get_data(lang)
    t = _read_file("pascal-api-example.pas.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Pascal API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Pascal API.

```pascal
{template}
```

Please refer to the [Pascal API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples)
for more details.

</details>
    """


def _go_api(lang):
    d = _get_data(lang)
    t = _read_file("go-api-example.go.in")
    env = jinja2.Environment()
    template = env.from_string(t).render(**d)

    return f"""
## Go API

<details>
<summary>Click to expand</summary>

You can use the following code to play with `supertonic-3` with Go API.

```go
{template}
```

Please refer to the [Go API documentation](https://github.com/k2-fsa/sherpa-onnx/tree/master/go-api-examples)
for more details.

</details>
    """


_SENTENCES = {
    "en": [
        "Hello world.",
        "How are you today?",
        "The sky is blue.",
        "I love machine learning.",
        "Python is awesome.",
        "Good morning everyone.",
        "Artificial intelligence is growing.",
        "Speech synthesis is fascinating.",
        "Neural networks are powerful.",
        "Text to speech converts text to audio.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps machines understand text.",
        "Deep learning has revolutionized artificial intelligence.",
        "Speech synthesis technology has advanced significantly.",
        "Neural voice cloning can replicate speaking styles.",
        "Text normalization is important for proper pronunciation.",
        "Voice assistants help us interact with technology naturally.",
        "Modern TTS systems use deep learning for high-quality speech.",
        "Human computer interaction has become more intuitive.",
    ],
    "es": [
        "Hola mundo.",
        "¿Cómo estás hoy?",
        "El cielo es azul.",
        "Me encanta el aprendizaje automático.",
        "Python es increíble.",
        "Buenos días a todos.",
        "La inteligencia artificial está creciendo.",
        "La síntesis de voz es fascinante.",
        "Las redes neuronales son poderosas.",
        "El texto a voz convierte texto en audio.",
        "El veloz marrón salta sobre el perro perezoso.",
        "El aprendizaje automático permite a las computadoras aprender.",
        "El procesamiento del lenguaje natural ayuda a las máquinas.",
        "El aprendizaje profundo ha revolucionado la inteligencia artificial.",
        "La tecnología de síntesis de voz ha avanzado significativamente.",
        "La clonación de voz neuronal puede replicar estilos de habla.",
        "La normalización de texto es importante para la pronunciación.",
        "Los asistentes de voz nos ayudan a interactuar con la tecnología.",
        "Los sistemas TTS modernos utilizan aprendizaje profundo.",
        "La interacción humano computadora se ha vuelto más intuitiva.",
    ],
    "pt": [
        "Olá mundo.",
        "Como você está hoje?",
        "O céu é azul.",
        "Eu amo aprendizado de máquina.",
        "Python é incrível.",
        "Bom dia a todos.",
        "A inteligência artificial está crescendo.",
        "A síntese de voz é fascinante.",
        "As redes neurais são poderosas.",
        "Texto para voz converte texto em áudio.",
        "A rápida raposa marrom salta sobre o cachorro preguiçoso.",
        "O aprendizado de máquina permite que computadores aprendam.",
        "O processamento de linguagem natural ajuda máquinas a entender.",
        "O aprendizado profundo revolucionou a inteligência artificial.",
        "A tecnologia de síntese de voz avançou significativamente.",
        "A clonagem de voz neural pode replicar estilos de fala.",
        "A normalização de texto é importante para pronúncia.",
        "Assistentes de voz nos ajudam a interagir com tecnologia.",
        "Sistemas TTS modernos usam aprendizado profundo para áudio.",
        "A interação humano computador tornou-se mais intuitiva.",
    ],
    "fr": [
        "Bonjour le monde.",
        "Comment allez-vous aujourd'hui?",
        "Le ciel est bleu.",
        "J'aime l'apprentissage automatique.",
        "Python est incroyable.",
        "Bonjour à tous.",
        "L'intelligence artificielle grandit.",
        "La synthèse vocale est fascinante.",
        "Les réseaux neuronaux sont puissants.",
        "Le texte en voix convertit le texte en audio.",
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "L'apprentissage automatique permet aux ordinateurs d'apprendre.",
        "Le traitement du langage naturel aide les machines à comprendre.",
        "L'apprentissage profond a révolutionné l'intelligence artificielle.",
        "La technologie de synthèse vocale a considérablement progressé.",
        "Le clonage vocal neuronal peut reproduire les styles de parole.",
        "La normalisation du texte est importante pour la prononciation.",
        "Les assistants vocaux nous aident à interagir avec la technologie.",
        "Les systèmes TTS modernes utilisent l'apprentissage profond.",
        "L'interaction homme machine est devenue plus intuitive.",
    ],
    "ko": [
        "안녕하세요 세계.",
        "오늘 어떻게 지내세요?",
        "하늘이 푸릅니다.",
        "기계학습을 사랑합니다.",
        "파이썬은 놀라워요.",
        "모든 분께 좋은 아침입니다.",
        "인공지능이 성장하고 있습니다.",
        "음성 합성은 매력적입니다.",
        "신경막은 강력합니다.",
        "텍스트 음성 변환이 텍스트를 오디오로 변환합니다.",
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "기계학습이 컴퓨터가 데이터로 학습할 수 있게 합니다.",
        "자연어 처리가 기계를 이해하도록 돕습니다.",
        "딥러닝이 인공지능을 혁신했습니다.",
        "음성 합성 기술이 크게 발전했습니다.",
        "음성 클로닝이 음성 스타일을 복제할 수 있습니다.",
        "텍스트 정규화가 올바른 발음에 중요합니다.",
        "음성 비서가 기술과 상호작용하는 데 도움이 됩니다.",
        "최신 TTS 시스템이 고품질 음성을 생성합니다.",
        "인간 컴퓨터 상호작용이 더 직관적이 되었습니다.",
    ],
    "ja": [
        "こんにちは世界。",
        "今日はどのように過ごしていますか。",
        "空は青く、風は穏やかです。",
        "機械学習はデータから学ぶ技術です。",
        "音声合成は文章を自然な声に変換します。",
        "図書館では多くの人が静かに本を読んでいます。",
        "新しい列車の時刻表は来週から使われます。",
        "研究者たちは小さな端末で動くモデルを評価しました。",
        "音声アシスタントは毎日の作業を手伝います。",
        "天気予報によると午後から雨が降るそうです。",
    ],
    "ar": [
        "مرحبا بالعالم.",
        "كيف حالك اليوم؟",
        "السماء زرقاء والهواء لطيف.",
        "يساعد التعلم الآلي الحواسيب على فهم البيانات.",
        "تحول تقنية تحويل النص إلى كلام الجمل إلى صوت واضح.",
        "قرأ الطلاب قصة قصيرة في المكتبة صباحا.",
        "أعلن القطار عن تأخير بسيط بسبب أعمال الصيانة.",
        "تعمل النماذج الصغيرة بسرعة على الأجهزة المحلية.",
        "يساعد المساعد الصوتي المستخدمين في المهام اليومية.",
        "تحتاج الأنظمة الحديثة إلى قراءة مستقرة للنصوص الطويلة.",
    ],
    "bg": [
        "Здравей свят.",
        "Как си днес?",
        "Небето е синьо, а вятърът е тих.",
        "Машинното обучение помага на компютрите да учат от данни.",
        "Синтезът на реч превръща текст в ясен звук.",
        "Учениците прочетоха кратка история в библиотеката.",
        "Влакът закъсня заради поддръжка на релсите.",
        "Малките модели работят бързо на локални устройства.",
        "Гласовите асистенти улесняват ежедневните задачи.",
        "Стабилното четене е важно за дълги и кратки изречения.",
    ],
    "cs": [
        "Ahoj světe.",
        "Jak se dnes máš?",
        "Obloha je modrá a vítr je mírný.",
        "Strojové učení pomáhá počítačům učit se z dat.",
        "Syntéza řeči převádí text na srozumitelný zvuk.",
        "Studenti četli krátký příběh v knihovně.",
        "Vlak měl zpoždění kvůli údržbě trati.",
        "Malé modely běží rychle na místních zařízeních.",
        "Hlasový asistent pomáhá s každodenními úkoly.",
        "Stabilní čtení je důležité pro dlouhé i krátké věty.",
    ],
    "da": [
        "Hej verden.",
        "Hvordan har du det i dag?",
        "Himlen er blå, og vinden er mild.",
        "Maskinlæring hjælper computere med at lære af data.",
        "Talesyntese omdanner tekst til tydelig lyd.",
        "Eleverne læste en kort historie på biblioteket.",
        "Toget blev forsinket på grund af sporarbejde.",
        "Små modeller kører hurtigt på lokale enheder.",
        "En stemmeassistent hjælper med daglige opgaver.",
        "Stabil oplæsning er vigtig for både korte og lange sætninger.",
    ],
    "de": [
        "Hallo Welt.",
        "Wie geht es dir heute?",
        "Der Himmel ist blau und der Wind ist mild.",
        "Maschinelles Lernen hilft Computern, aus Daten zu lernen.",
        "Sprachsynthese wandelt Text in klare Sprache um.",
        "Die Schüler lasen am Morgen eine kurze Geschichte.",
        "Der Zug hatte wegen Wartungsarbeiten Verspätung.",
        "Kleine Modelle laufen schnell auf lokalen Geräten.",
        "Ein Sprachassistent hilft bei alltäglichen Aufgaben.",
        "Stabiles Vorlesen ist für kurze und lange Texte wichtig.",
    ],
    "el": [
        "Γεια σου κόσμε.",
        "Πώς είσαι σήμερα;",
        "Ο ουρανός είναι γαλάζιος και ο άνεμος ήρεμος.",
        "Η μηχανική μάθηση βοηθά τους υπολογιστές να μαθαίνουν από δεδομένα.",
        "Η σύνθεση ομιλίας μετατρέπει το κείμενο σε καθαρό ήχο.",
        "Οι μαθητές διάβασαν μια μικρή ιστορία στη βιβλιοθήκη.",
        "Το τρένο καθυστέρησε λόγω εργασιών συντήρησης.",
        "Τα μικρά μοντέλα λειτουργούν γρήγορα σε τοπικές συσκευές.",
        "Ο φωνητικός βοηθός διευκολύνει τις καθημερινές εργασίες.",
        "Η σταθερή ανάγνωση είναι σημαντική για σύντομα και μεγάλα κείμενα.",
    ],
    "et": [
        "Tere maailm.",
        "Kuidas sul täna läheb?",
        "Taevas on sinine ja tuul on vaikne.",
        "Masinõpe aitab arvutitel andmetest õppida.",
        "Kõnesüntees muudab teksti selgeks heliks.",
        "Õpilased lugesid raamatukogus lühikest lugu.",
        "Rong hilines rööbaste hoolduse tõttu.",
        "Väikesed mudelid töötavad kiiresti kohalikes seadmetes.",
        "Häälassistent aitab igapäevaste ülesannetega.",
        "Stabiilne lugemine on tähtis nii lühikeste kui pikkade lausete jaoks.",
    ],
    "fi": [
        "Hei maailma.",
        "Miten voit tänään?",
        "Taivas on sininen ja tuuli on lempeä.",
        "Koneoppiminen auttaa tietokoneita oppimaan datasta.",
        "Puhesynteesi muuttaa tekstin selkeäksi ääneksi.",
        "Oppilaat lukivat lyhyen tarinan kirjastossa.",
        "Juna myöhästyi raiteiden huollon vuoksi.",
        "Pienet mallit toimivat nopeasti paikallisilla laitteilla.",
        "Ääniavustaja auttaa päivittäisissä tehtävissä.",
        "Vakaa lukeminen on tärkeää sekä lyhyille että pitkille lauseille.",
    ],
    "hi": [
        "नमस्ते दुनिया.",
        "आज आप कैसे हैं?",
        "आसमान नीला है और हवा हल्की है.",
        "मशीन लर्निंग कंप्यूटरों को डेटा से सीखने में मदद करती है.",
        "वाक् संश्लेषण पाठ को स्पष्ट ध्वनि में बदलता है.",
        "छात्रों ने पुस्तकालय में एक छोटी कहानी पढ़ी.",
        "पटरियों की मरम्मत के कारण ट्रेन थोड़ी देर से आई.",
        "छोटे मॉडल स्थानीय उपकरणों पर तेज़ी से चलते हैं.",
        "वॉयस असिस्टेंट रोज़मर्रा के कामों में मदद करता है.",
        "लंबे और छोटे वाक्यों के लिए स्थिर पढ़ना महत्वपूर्ण है.",
    ],
    "hr": [
        "Pozdrav svijete.",
        "Kako si danas?",
        "Nebo je plavo, a vjetar je blag.",
        "Strojno učenje pomaže računalima učiti iz podataka.",
        "Sinteza govora pretvara tekst u jasan zvuk.",
        "Učenici su u knjižnici pročitali kratku priču.",
        "Vlak je kasnio zbog održavanja pruge.",
        "Mali modeli brzo rade na lokalnim uređajima.",
        "Glasovni asistent pomaže u svakodnevnim zadacima.",
        "Stabilno čitanje važno je za kratke i duge rečenice.",
    ],
    "hu": [
        "Helló világ.",
        "Hogy vagy ma?",
        "Az ég kék, a szél pedig enyhe.",
        "A gépi tanulás segít a számítógépeknek adatokból tanulni.",
        "A beszédszintézis a szöveget tiszta hanggá alakítja.",
        "A diákok rövid történetet olvastak a könyvtárban.",
        "A vonat a pálya karbantartása miatt késett.",
        "A kis modellek gyorsan futnak helyi eszközökön.",
        "A hangasszisztens segít a mindennapi feladatokban.",
        "A stabil felolvasás fontos rövid és hosszú mondatoknál is.",
    ],
    "id": [
        "Halo dunia.",
        "Apa kabar hari ini?",
        "Langit berwarna biru dan angin terasa lembut.",
        "Pembelajaran mesin membantu komputer belajar dari data.",
        "Sintesis ucapan mengubah teks menjadi suara yang jelas.",
        "Para siswa membaca cerita pendek di perpustakaan.",
        "Kereta terlambat karena perawatan rel.",
        "Model kecil berjalan cepat di perangkat lokal.",
        "Asisten suara membantu pekerjaan sehari-hari.",
        "Pembacaan yang stabil penting untuk kalimat pendek dan panjang.",
    ],
    "it": [
        "Ciao mondo.",
        "Come stai oggi?",
        "Il cielo è blu e il vento è leggero.",
        "L'apprendimento automatico aiuta i computer a imparare dai dati.",
        "La sintesi vocale trasforma il testo in audio chiaro.",
        "Gli studenti hanno letto una breve storia in biblioteca.",
        "Il treno ha subito un ritardo per lavori sui binari.",
        "I modelli piccoli funzionano rapidamente sui dispositivi locali.",
        "Un assistente vocale aiuta nelle attività quotidiane.",
        "Una lettura stabile è importante per frasi brevi e lunghe.",
    ],
    "lt": [
        "Labas pasauli.",
        "Kaip šiandien laikaisi?",
        "Dangus mėlynas, o vėjas švelnus.",
        "Mašininis mokymasis padeda kompiuteriams mokytis iš duomenų.",
        "Kalbos sintezė paverčia tekstą aiškiu garsu.",
        "Mokiniai bibliotekoje perskaitė trumpą istoriją.",
        "Traukinys vėlavo dėl bėgių priežiūros.",
        "Maži modeliai greitai veikia vietiniuose įrenginiuose.",
        "Balso asistentas padeda atlikti kasdienes užduotis.",
        "Stabilus skaitymas svarbus trumpiems ir ilgiems sakiniams.",
    ],
    "lv": [
        "Sveika pasaule.",
        "Kā tev šodien klājas?",
        "Debesis ir zilas, un vējš ir maigs.",
        "Mašīnmācīšanās palīdz datoriem mācīties no datiem.",
        "Runas sintēze pārvērš tekstu skaidrā skaņā.",
        "Skolēni bibliotēkā lasīja īsu stāstu.",
        "Vilciens kavējās sliežu remonta dēļ.",
        "Mazie modeļi ātri darbojas vietējās ierīcēs.",
        "Balss asistents palīdz ikdienas uzdevumos.",
        "Stabila lasīšana ir svarīga īsiem un gariem teikumiem.",
    ],
    "nl": [
        "Hallo wereld.",
        "Hoe gaat het vandaag?",
        "De lucht is blauw en de wind is zacht.",
        "Machine learning helpt computers om van gegevens te leren.",
        "Spraaksynthese zet tekst om in duidelijke audio.",
        "De leerlingen lazen een kort verhaal in de bibliotheek.",
        "De trein had vertraging door onderhoud aan het spoor.",
        "Kleine modellen draaien snel op lokale apparaten.",
        "Een stemassistent helpt bij dagelijkse taken.",
        "Stabiel voorlezen is belangrijk voor korte en lange zinnen.",
    ],
    "pl": [
        "Witaj świecie.",
        "Jak się dziś masz?",
        "Niebo jest niebieskie, a wiatr jest łagodny.",
        "Uczenie maszynowe pomaga komputerom uczyć się z danych.",
        "Synteza mowy zamienia tekst w wyraźny dźwięk.",
        "Uczniowie przeczytali krótką historię w bibliotece.",
        "Pociąg spóźnił się z powodu konserwacji torów.",
        "Małe modele działają szybko na lokalnych urządzeniach.",
        "Asystent głosowy pomaga w codziennych zadaniach.",
        "Stabilne czytanie jest ważne dla krótkich i długich zdań.",
    ],
    "ro": [
        "Salut lume.",
        "Cum te simți astăzi?",
        "Cerul este albastru, iar vântul este blând.",
        "Învățarea automată ajută computerele să învețe din date.",
        "Sinteza vocală transformă textul în sunet clar.",
        "Elevii au citit o poveste scurtă la bibliotecă.",
        "Trenul a întârziat din cauza lucrărilor la șine.",
        "Modelele mici rulează rapid pe dispozitive locale.",
        "Asistentul vocal ajută la sarcinile zilnice.",
        "Citirea stabilă este importantă pentru propoziții scurte și lungi.",
    ],
    "ru": [
        "Привет мир.",
        "Как у тебя дела сегодня?",
        "Небо голубое, а ветер мягкий.",
        "Машинное обучение помогает компьютерам учиться на данных.",
        "Синтез речи превращает текст в понятный звук.",
        "Ученики прочитали короткий рассказ в библиотеке.",
        "Поезд задержался из-за ремонта путей.",
        "Небольшие модели быстро работают на локальных устройствах.",
        "Голосовой помощник помогает в повседневных задачах.",
        "Стабильное чтение важно для коротких и длинных предложений.",
    ],
    "sk": [
        "Ahoj svet.",
        "Ako sa dnes máš?",
        "Obloha je modrá a vietor je mierny.",
        "Strojové učenie pomáha počítačom učiť sa z dát.",
        "Syntéza reči premieňa text na zrozumiteľný zvuk.",
        "Žiaci čítali krátky príbeh v knižnici.",
        "Vlak meškal pre údržbu trate.",
        "Malé modely bežia rýchlo na lokálnych zariadeniach.",
        "Hlasový asistent pomáha s každodennými úlohami.",
        "Stabilné čítanie je dôležité pre krátke aj dlhé vety.",
    ],
    "sl": [
        "Pozdravljen svet.",
        "Kako si danes?",
        "Nebo je modro in veter je nežen.",
        "Strojno učenje pomaga računalnikom učiti se iz podatkov.",
        "Sinteza govora pretvori besedilo v jasen zvok.",
        "Učenci so v knjižnici prebrali kratko zgodbo.",
        "Vlak je zamujal zaradi vzdrževanja tirov.",
        "Majhni modeli hitro delujejo na lokalnih napravah.",
        "Glasovni pomočnik pomaga pri vsakodnevnih opravilih.",
        "Stabilno branje je pomembno za kratke in dolge stavke.",
    ],
    "sv": [
        "Hej världen.",
        "Hur mår du idag?",
        "Himlen är blå och vinden är mild.",
        "Maskininlärning hjälper datorer att lära sig av data.",
        "Talsyntes omvandlar text till tydligt ljud.",
        "Eleverna läste en kort berättelse på biblioteket.",
        "Tåget blev försenat på grund av spårunderhåll.",
        "Små modeller kör snabbt på lokala enheter.",
        "En röstassistent hjälper till med vardagliga uppgifter.",
        "Stabil uppläsning är viktig för korta och långa meningar.",
    ],
    "tr": [
        "Merhaba dünya.",
        "Bugün nasılsın?",
        "Gökyüzü mavi ve rüzgar hafif.",
        "Makine öğrenimi bilgisayarların verilerden öğrenmesine yardımcı olur.",
        "Konuşma sentezi metni anlaşılır sese dönüştürür.",
        "Öğrenciler kütüphanede kısa bir hikaye okudu.",
        "Tren ray bakımı nedeniyle gecikti.",
        "Küçük modeller yerel cihazlarda hızlı çalışır.",
        "Sesli asistan günlük işlerde yardımcı olur.",
        "Kararlı okuma kısa ve uzun cümleler için önemlidir.",
    ],
    "uk": [
        "Привіт світе.",
        "Як ти сьогодні?",
        "Небо блакитне, а вітер лагідний.",
        "Машинне навчання допомагає комп'ютерам вчитися на даних.",
        "Синтез мовлення перетворює текст на зрозумілий звук.",
        "Учні прочитали коротку історію в бібліотеці.",
        "Потяг затримався через ремонт колії.",
        "Невеликі моделі швидко працюють на локальних пристроях.",
        "Голосовий помічник допомагає з щоденними завданнями.",
        "Стабільне читання важливе для коротких і довгих речень.",
    ],
    "vi": [
        "Xin chào thế giới.",
        "Hôm nay bạn thế nào?",
        "Bầu trời xanh và gió rất nhẹ.",
        "Học máy giúp máy tính học từ dữ liệu.",
        "Tổng hợp giọng nói chuyển văn bản thành âm thanh rõ ràng.",
        "Học sinh đọc một câu chuyện ngắn trong thư viện.",
        "Tàu bị trễ vì công việc bảo trì đường ray.",
        "Các mô hình nhỏ chạy nhanh trên thiết bị cục bộ.",
        "Trợ lý giọng nói hỗ trợ các công việc hằng ngày.",
        "Việc đọc ổn định rất quan trọng cho câu ngắn và câu dài.",
    ],
}


def _samples(lang):
    sentences = _SENTENCES[lang]
    s = """
## Samples

sample audios for different speakers are listed below:

"""
    for sid in range(10):
        s += f"\n### Speaker {sid}\n"
        for i, text in enumerate(sentences):
            s += f"""
#### {i}

> {text}

<audio controls>
  <source src="/sherpa/onnx/tts/all/supertonic/v3/mp3/{lang}/sid-{sid}-{lang}-{i}.mp3" type="audio/mp3">
</audio>

"""

    return s


def generate_supertonic_v3():
    """Generate markdown files for supertonic v3 TTS model for all 31 languages.

    Returns a list of (lang_code, lang_name, filename) tuples for use in SUMMARY.md.
    """
    results = []

    for lang in _LANG_INFO:
        lang_name = _LANG_INFO[lang][0]
        filename = f"supertonic-3-{lang}"

        s = f"""
# supertonic-3-{lang}

||||||
|---|---|---|---|---|
|[Info about this model](#info-about-this-model)|[Download the model](#download-the-model)|[Android APK](#android-apk)|[Python API](#python-api)|[C API](#c-api)|
|[C++ API](#c-api-1)|[Rust API](#rust-api)|[Node.js API](#nodejs-addon-api)|[Dart API](#dart-api)|[Swift API](#swift-api)|
|[C# API](#c-api-2)|[Kotlin API](#kotlin-api)|[Java API](#java-api)|[Pascal API](#pascal-api)|[Go API](#go-api)|
|[Samples](#samples)|||||

## Info about this model

This model is supertonic 3 from <https://huggingface.co/Supertone/supertonic-3>

It supports **31 languages**: en, ko, ja, ar, bg, cs, da, de, el, es, et, fi, fr, hi, hr, hu, id, it, lt, lv, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi.

This page shows samples for **{lang_name}** (`{lang}`).

| Number of speakers | Sample rate |
|--------------------|-------------|
| 10 | 24000|

### Speaker IDs

| sid | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-----|---|---|---|---|---|---|---|---|---|---|

"""
        s += _download()
        s += _android_apk(lang)
        s += _python_api(lang)
        s += _c_api(lang)
        s += _cxx_api(lang)
        s += _rust_api(lang)
        s += _node_addon_api(lang)
        s += _dart_api(lang)
        s += _swift_api(lang)
        s += _csharp_api(lang)
        s += _kotlin_api(lang)
        s += _java_api(lang)
        s += _pascal_api(lang)
        s += _go_api(lang)
        s += _samples(lang)

        out_path = f"book/src/{lang_name}/{filename}.md"
        with open(out_path, "w") as f:
            f.write(s)

        results.append((lang, lang_name, filename))

    return results
