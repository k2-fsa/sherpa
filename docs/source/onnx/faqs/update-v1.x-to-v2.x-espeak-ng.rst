.. _update_v1_to_v2_espeak_ng:

How to use espeak-ng with sherpa-onnx v2.x
==========================================

`sherpa-onnx`_ v2.x has removed `espeak-ng`_ and `piper-phonemize`_ from its
code base, see `<https://github.com/k2-fsa/sherpa-onnx/pull/4055>`_. This page
shows how to update your code from v1.x to v2.x if your TTS model needs
`espeak-ng`_ for phonemization.

.. contents:: On this page
   :local:
   :depth: 2

What changed
------------

In v1.x, `espeak-ng`_ and `piper-phonemize`_ were built into `sherpa-onnx`_.
For a TTS model using them (e.g., a `piper`_ VITS model), you downloaded an
``espeak-ng-data`` directory and passed it via ``--vits-data-dir`` (or the
``data_dir`` field of a model config) and `sherpa-onnx`_ phonemized the input
text internally.

In v2.x:

- The text frontends of `sherpa-onnx`_ are now

  - **lexicon** — built in, no extra dependency
  - **phonemize** — you phonemize the text outside of `sherpa-onnx`_, e.g.,
    with `piper-phonemize`_, and pass the phonemized result to `sherpa-onnx`_.

- ``data_dir``, e.g., ``--vits-data-dir``, is deprecated and ignored. A
  warning is printed if you still pass it.

- `piper-phonemize`_ is GPL because of `espeak-ng`_, so it stays optional and
  is not bundled with `sherpa-onnx`_. `sherpa-onnx`_ itself is GPL-free now.

The error you get in v2.x
-------------------------

If you run a model that needs `espeak-ng`_ without a lexicon and without
phonemizing the text first, you get:

.. code-block:: text

   phoneme_codepoints is empty, no lexicon, and no frontend available.
   Please provide phoneme_codepoints via GenerationConfig or a lexicon via --vits-lexicon.

Please choose one of the two options below.

Option 1: Use a lexicon
-----------------------

If there is a lexicon for your language, nothing changes for you except that
you pass the lexicon instead of ``espeak-ng-data``. For instance, for English
`piper`_ models, you can use ``lexicon-en-us.txt`` from the
`tts-models <https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_
release. No external phonemizer is needed.

Please see
`<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-piper-lexicon.py>`_

Option 2: Phonemize the text with piper-phonemize
-------------------------------------------------

Install `piper-phonemize`_ for your favorite language, phonemize the text
outside of `sherpa-onnx`_, and pass the result in. The examples below use the
Python API; please see the next section for other languages.

Install `piper-phonemize`_::

   pip install piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

Download a model::

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
   tar xf vits-piper-en_US-amy-low.tar.bz2

.. code-block:: python

   import sherpa_onnx
   from piper_phonemize import phonemize_espeak

   tts_config = sherpa_onnx.OfflineTtsConfig(
       model=sherpa_onnx.OfflineTtsModelConfig(
           vits=sherpa_onnx.OfflineTtsVitsModelConfig(
               model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
               tokens="./vits-piper-en_US-amy-low/tokens.txt",
           ),
           debug=True,
       ),
   )
   tts = sherpa_onnx.OfflineTts(tts_config)

   # The espeak-ng voice for the phonemizer is taken from the model metadata.
   # Fall back to en-us if the model has no language information.
   lang = tts.lang
   if not lang:
       lang = "en-us"

   text = "Today as always, men fall into two groups: slaves and free men."
   sentences = phonemize_espeak(text, lang)
   phoneme_codepoints = [[ord(c) for c in sentence] for sentence in sentences]

   gen_config = sherpa_onnx.GenerationConfig()
   gen_config.sid = 0
   gen_config.speed = 1.0
   gen_config.phoneme_codepoints = phoneme_codepoints

   # The text has been phonemized already, so pass an empty string here.
   audio = tts.generate("", gen_config)

Please see
`<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-piper-phonemize.py>`_
for a complete example. There are more examples for other model families:

  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-matcha-en-phonemize.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-kokoro-en-phonemize.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-kitten-phonemize.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-zipvoice-phonemize.py>`_
  - `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-coqui-phonemize.py>`_

Other language bindings
-----------------------

`piper-phonemize`_ supports many languages. The idea is the same in every
language:

1. Install `piper-phonemize`_ for your language.
2. Phonemize the text with it, using the espeak-ng voice of your model.
3. Pass the phoneme codepoints to `sherpa-onnx`_ via ``GenerationConfig`` and
   generate with an empty text.

All examples below use the ``vits-piper-en_US-amy-low`` model from the
beginning of this page:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
   tar xf vits-piper-en_US-amy-low.tar.bz2

The ``espeak-ng-data`` directory shipped inside the model package is used only
by the phonemizer; `sherpa-onnx`_ itself does not read it.

C/C++
^^^^^

Step 1. Install `piper-phonemize`_ from a pre-built release or build it from
source:

.. code-block:: bash

   git clone https://github.com/csukuangfj/piper-phonemize
   cd piper-phonemize
   cmake -Bbuild -DBUILD_SHARED_LIBS=ON
   cmake --build build --config Release

Step 2. Link your app with both ``piper-phonemize`` and `sherpa-onnx`_ and
phonemize the text before generating:

.. code-block:: cpp

   #include <cstdio>
   #include <string>

   #include "piper-phonemize/cxx-api.h"
   #include "sherpa-onnx/c-api/cxx-api.h"

   namespace pp = piper_phonemize::cxx;

   int main() {
     using namespace sherpa_onnx::cxx;  // NOLINT

     // espeak-ng-data is used only by piper-phonemize; sherpa-onnx itself
     // does not use espeak-ng anymore.
     if (pp::Initialize("./vits-piper-en_US-amy-low/espeak-ng-data") < 0) {
       fprintf(stderr, "Failed to initialize piper-phonemize\n");
       return -1;
     }

     OfflineTtsConfig config;
     config.model.vits.model = "./vits-piper-en_US-amy-low/en_US-amy-low.onnx";
     config.model.vits.tokens = "./vits-piper-en_US-amy-low/tokens.txt";
     config.model.num_threads = 1;
     config.model.debug = true;
     auto tts = OfflineTts::Create(config);

     std::string text =
         "Today as always, men fall into two groups: slaves and free men.";

     // The espeak-ng voice for the phonemizer is taken from the model
     // metadata. Fall back to en-us if the model has no language information.
     std::string lang = tts.Lang();
     if (lang.empty()) {
       lang = "en-us";
     }

     auto result = pp::Phonemize(text, lang);

     GenerationConfig gen_config;
     gen_config.sid = 0;
     gen_config.speed = 1.0;
     gen_config.phoneme_codepoints = result.sentences;

     // The text has been phonemized already; pass an empty string here.
     GeneratedAudio audio = tts.Generate("", gen_config);
     WriteWave("./generated.wav", {audio.samples, audio.sample_rate});
     return 0;
   }

If you use the C API, phonemize with ``PiperPhonemizeInitialize()`` and
``PiperPhonemizeText()`` from ``piper-phonemize/c-api.h`` and fill
``phoneme_codepoints``, ``phoneme_codepoints_lens``, and
``phoneme_codepoints_num_sentences`` of ``SherpaOnnxGenerationConfig``
(``sherpa-onnx/c-api/c-api.h``).

Java/Kotlin
^^^^^^^^^^^

Step 1. Add the `piper-phonemize`_ artifacts from JitPack. Add the
``jitpack.io`` repository and the following dependencies to your ``pom.xml``
(for Gradle, use the same coordinates in ``build.gradle``):

.. code-block:: xml

   <repositories>
     <repository>
       <id>jitpack.io</id>
       <url>https://jitpack.io</url>
     </repository>
   </repositories>

   <dependencies>
     <!-- JVM core API -->
     <dependency>
       <groupId>com.github.csukuangfj.piper-phonemize</groupId>
       <artifactId>piper-phonemize-jvm</artifactId>
       <version>v1.5.9</version>
     </dependency>

     <!-- native lib, pick ONE for your target platform -->
     <dependency>
       <groupId>com.github.csukuangfj.piper-phonemize</groupId>
       <artifactId>piper-phonemize-native-lib-osx-aarch64</artifactId>
       <version>v1.5.9</version>
     </dependency>
   </dependencies>

Note that `piper-phonemize`_ has no all-in-one package: the JVM API and the
native libraries are separate artifacts, so both dependencies are needed.

The native artifact is one of ``piper-phonemize-native-lib-osx-aarch64``,
``piper-phonemize-native-lib-osx-x64``,
``piper-phonemize-native-lib-linux-x64``,
``piper-phonemize-native-lib-linux-aarch64``,
``piper-phonemize-native-lib-win-x64``, or
``piper-phonemize-native-lib-win-arm64``.

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_. In Java:

.. code-block:: java

   import com.github.csukuangfj.piper.phonemize.PiperPhonemize;
   import com.k2fsa.sherpa.onnx.*;

   public class Main {
     public static void main(String[] args) {
       OfflineTtsVitsModelConfig vits =
           OfflineTtsVitsModelConfig.builder()
               .setModel("./vits-piper-en_US-amy-low/en_US-amy-low.onnx")
               .setTokens("./vits-piper-en_US-amy-low/tokens.txt")
               .build();
       OfflineTtsConfig config =
           OfflineTtsConfig.builder()
               .setModel(
                   OfflineTtsModelConfig.builder()
                       .setVits(vits)
                       .setNumThreads(1)
                       .setDebug(true)
                       .build())
               .build();
       OfflineTts tts = new OfflineTts(config);

       // initialize() extracts the espeak-ng-data embedded in piper-phonemize.
       // That data feeds only the phonemizer; sherpa-onnx does not use
       // espeak-ng anymore.
       PiperPhonemize.initialize();

       String text =
           "Today as always, men fall into two groups: slaves and free men.";

       // The espeak-ng voice for the phonemizer is taken from the model
       // metadata. Fall back to en-us if the model has no language information.
       String voice = tts.getLang();
       if (voice == null || voice.isEmpty()) {
         voice = "en-us";
       }

       int[][] phonemeCodepoints;
       try (PiperPhonemize.Result result = PiperPhonemize.text(text, voice)) {
         phonemeCodepoints = new int[result.getNumSentences()][];
         for (int i = 0; i < result.getNumSentences(); i++) {
           phonemeCodepoints[i] = result.getPhonemes(i);
         }
       }

       GenerationConfig genConfig = new GenerationConfig();
       genConfig.setSid(0);
       genConfig.setSpeed(1.0f);
       genConfig.setPhonemeCodepoints(phonemeCodepoints);

       // The text has been phonemized already; pass an empty string here.
       GeneratedAudio audio = tts.generateWithConfig("", genConfig);
       audio.save("tts-piper-en-phonemize.wav");
       tts.release();
     }
   }

In Kotlin:

.. code-block:: kotlin

   import com.github.csukuangfj.piper.phonemize.PiperPhonemize

   // Phonemize the text and return the phoneme codepoints, one inner array
   // per sentence.
   fun phonemize(tts: OfflineTts, text: String): Array<IntArray> {
     PiperPhonemize.initialize()
     var voice = tts.lang()
     if (voice.isEmpty()) {
       voice = "en-us"
     }
     val result =
       PiperPhonemize.text(text, voice)
         ?: throw RuntimeException("Failed to phonemize the text")
     result.use {
       return Array(it.numSentences) { i -> it.getPhonemes(i) }
     }
   }

   val genConfig = GenerationConfig(phonemeCodepoints = phonemize(tts, text))
   // The text has been phonemized already; pass an empty string here.
   val audio = tts.generateWithConfig(text = "", config = genConfig)
   audio.save(filename = "tts-piper-en-phonemize.wav")

Go
^^

Step 1. Add the dependencies:

.. code-block:: bash

   go get github.com/csukuangfj/piper-phonemize-go/piper_phonemize
   go get github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: go

   package main

   import (
       "log"

       pp "github.com/csukuangfj/piper-phonemize-go/piper_phonemize"
       sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
   )

   func main() {
       config := sherpa.OfflineTtsConfig{}
       config.Model.Vits.Model = "./vits-piper-en_US-amy-low/en_US-amy-low.onnx"
       config.Model.Vits.Tokens = "./vits-piper-en_US-amy-low/tokens.txt"
       config.Model.NumThreads = 1
       config.Model.Debug = 1

       tts := sherpa.NewOfflineTts(&config)
       defer sherpa.DeleteOfflineTts(tts)

       // Initialize("") uses the espeak-ng-data embedded in
       // piper-phonemize-go. It has nothing to do with sherpa-onnx;
       // sherpa-onnx itself does not use espeak-ng anymore.
       if ret := pp.Initialize(""); ret < 0 {
           log.Fatalf("Failed to initialize piper-phonemize")
       }

       text := "Today as always, men fall into two groups: slaves and free men."

       // The espeak-ng voice for the phonemizer is taken from the model
       // metadata. Fall back to en-us if the model has no language
       // information.
       lang := tts.Lang()
       if lang == "" {
           lang = "en-us"
       }

       result := pp.Phonemize(text, lang)
       if result == nil {
           log.Fatalf("Failed to phonemize the text")
       }
       defer pp.DeletePhonemizeResult(result)

       var sentences [][]int32
       for i := 0; i < result.GetNumSentences(); i++ {
           var s []int32
           for _, cp := range result.GetPhonemes(i) {
               s = append(s, int32(cp))
           }
           sentences = append(sentences, s)
       }

       cfg := sherpa.GenerationConfig{
           SilenceScale:      0.2,
           Speed:             1.0,
           Sid:               0,
           PhonemeCodepoints: sentences,
       }
       // The text has been phonemized already; pass an empty string here.
       audio := tts.GenerateWithConfig("", &cfg, nil)
       audio.Save("./generated.wav")
   }

C#/.NET
^^^^^^^

Step 1. Add the NuGet packages to your ``.csproj``:

.. code-block:: xml

   <ItemGroup>
     <PackageReference Include="org.k2fsa.sherpa.onnx" Version="*" />
     <PackageReference Include="com.github.csukuangfj.piper.phonemize" Version="*" />
   </ItemGroup>

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: csharp

   using PiperPhonemize;
   using SherpaOnnx;

   // The piper-phonemize NuGet package copies espeak-ng-data next to the
   // executable. It feeds only the phonemizer; sherpa-onnx does not use
   // espeak-ng anymore.
   string dataDir = Path.Combine(AppContext.BaseDirectory, "espeak-ng-data");
   if (PiperPhonemizeApi.Initialize(dataDir) < 0) {
     throw new Exception("Failed to initialize piper-phonemize");
   }

   var config = new OfflineTtsConfig();
   config.Model.Vits.Model = "./vits-piper-en_US-amy-low/en_US-amy-low.onnx";
   config.Model.Vits.Tokens = "./vits-piper-en_US-amy-low/tokens.txt";
   config.Model.NumThreads = 1;
   config.Model.Debug = 1;
   var tts = new OfflineTts(config);

   string text = "Today as always, men fall into two groups: slaves and free men.";

   // The espeak-ng voice for the phonemizer is taken from the model metadata.
   // Fall back to en-us if the model has no language information.
   string lang = string.IsNullOrEmpty(tts.Lang) ? "en-us" : tts.Lang;

   int[][] phonemeCodepoints;
   using (var result = PiperPhonemizeApi.Text(text, lang)) {
     phonemeCodepoints = new int[result.NumSentences][];
     for (int i = 0; i < result.NumSentences; i++) {
       string ipa = result.GetPhonemesAsString(i) ?? "";
       int[] cps = new int[ipa.Length];
       for (int j = 0; j < ipa.Length; j++) {
         cps[j] = ipa[j];
       }
       phonemeCodepoints[i] = cps;
     }
   }

   var genConfig = new OfflineTtsGenerationConfig();
   genConfig.Sid = 0;
   genConfig.Speed = 1.0f;
   genConfig.PhonemeCodepoints = phonemeCodepoints;

   // The text has been phonemized already; pass an empty string here.
   var audio = tts.GenerateWithConfig("", genConfig, null);
   audio.SaveToWaveFile("./generated-dotnet-phonemize.wav");

Rust
^^^^

Step 1. Add the crates to your ``Cargo.toml``. The ``piper-phonemize`` crate
downloads pre-built libraries and embeds ``espeak-ng-data`` automatically:

.. code-block:: toml

   [dependencies]
   sherpa-onnx = "2.0.1"
   piper-phonemize = "1.5.9"

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: rust

   use sherpa_onnx::{
       GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsVitsModelConfig,
   };

   fn main() {
       // Note: the espeak-ng-data used by piper-phonemize is embedded in the
       // crate and initialized automatically on first use. It has nothing to
       // do with sherpa-onnx; sherpa-onnx itself does not use espeak-ng
       // anymore.

       let config = OfflineTtsConfig {
           model: sherpa_onnx::OfflineTtsModelConfig {
               vits: OfflineTtsVitsModelConfig {
                   model: Some("./vits-piper-en_US-amy-low/en_US-amy-low.onnx".into()),
                   tokens: Some("./vits-piper-en_US-amy-low/tokens.txt".into()),
                   ..Default::default()
               },
               num_threads: 2,
               debug: false,
               ..Default::default()
           },
           ..Default::default()
       };

       let tts = OfflineTts::create(&config).expect("Failed to create OfflineTts");
       let text = "Today as always, men fall into two groups: slaves and free men.";

       // The espeak-ng voice for the phonemizer is taken from the model
       // metadata. Fall back to en-us if the model has no language information.
       let lang = {
           let l = tts.lang();
           if l.is_empty() {
               "en-us".to_string()
           } else {
               l
           }
       };

       let result =
           piper_phonemize::phonemize(text, &lang).expect("Failed to phonemize the text");

       let mut phoneme_codepoints = Vec::new();
       for i in 0..result.num_sentences() {
           let phonemes = result.get_phonemes(i).unwrap_or_default();
           phoneme_codepoints.push(phonemes.into_iter().map(|c| c as i32).collect::<Vec<i32>>());
       }

       let gen_config = GenerationConfig {
           sid: 0,
           speed: 1.0,
           phoneme_codepoints: Some(phoneme_codepoints),
           ..Default::default()
       };

       // The text has been phonemized already; pass an empty string here.
       let audio = tts
           .generate_with_config(
               "",
               &gen_config,
               Some(|_samples: &[f32], progress: f32| -> bool {
                   println!("Progress: {:.1}%", progress * 100.0);
                   true
               }),
           )
           .expect("Generation failed");

       audio.save("./generated-vits-en-phonemize-rust.wav");
   }

JavaScript (Node.js)
^^^^^^^^^^^^^^^^^^^^

Step 1. Install the packages. ``piper-phonemize-node`` is the native addon;
``npm install piper-phonemize`` installs the WASM version, which exposes the
same functions:

.. code-block:: bash

   npm install sherpa-onnx-node piper-phonemize-node

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');
   const piperPhonemize = require('piper-phonemize-node');

   const tts = new sherpa_onnx.OfflineTts({
     model: {
       vits: {
         model: './vits-piper-en_US-amy-low/en_US-amy-low.onnx',
         tokens: './vits-piper-en_US-amy-low/tokens.txt',
       },
       debug: true,
       numThreads: 1,
       provider: 'cpu',
     },
     maxNumSentences: 1,
   });

   const text =
       'Today as always, men fall into two groups: slaves and free men.';

   // The espeak-ng voice for the phonemizer is taken from the model metadata.
   // Fall back to en-us if the model has no language information.
   const voice = tts.lang || 'en-us';
   const phonemeCodepoints = piperPhonemize.phonemize(text, voice)
       // piper may emit empty sentences (e.g., a trailing segment after
       // punctuation); the C API requires every sentence to be non-empty.
       .filter((sentence) => sentence.length > 0);

   const genConfig = new sherpa_onnx.GenerationConfig({
     phonemeCodepoints,
     sid: 0,
     speed: 1.0,
     silenceScale: 0.2,
   });

   // The text has been phonemized already; pass an empty string here.
   const audio = tts.generate({text: '', generationConfig: genConfig});

   sherpa_onnx.writeWave(
       'test-piper-en-phonemize.wav',
       {samples: audio.samples, sampleRate: audio.sampleRate});

Dart/Flutter
^^^^^^^^^^^^

Step 1. Add the packages to your ``pubspec.yaml``:

.. code-block:: yaml

   dependencies:
     sherpa_onnx: ^2.0.1
     piper_phonemize: ^1.5.9

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: dart

   import 'package:piper_phonemize/piper_phonemize.dart' as piper;
   import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

   Future<void> main() async {
     await sherpa_onnx.initBindingsAsync();
     await piper.initBindingsAsync();

     final model = './vits-piper-en_US-amy-low/en_US-amy-low.onnx';
     final tokens = './vits-piper-en_US-amy-low/tokens.txt';

     // espeak-ng-data feeds only the piper phonemizer; sherpa-onnx ignores
     // it. Use the copy shipped inside the model package.
     final ret = piper.PiperPhonemize
         .initialize('./vits-piper-en_US-amy-low/espeak-ng-data');
     if (ret < 0) {
       throw Exception('Failed to initialize the piper phonemizer');
     }

     final tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
       model: sherpa_onnx.OfflineTtsModelConfig(
         vits: sherpa_onnx.OfflineTtsVitsModelConfig(
           model: model,
           tokens: tokens,
         ),
         numThreads: 1,
         debug: true,
       ),
       maxNumSentences: 1,
     ));

     const text =
         'Today as always, men fall into two groups: slaves and free men.';

     // The espeak-ng voice for the phonemizer is taken from the model
     // metadata. Fall back to en-us if the model has no language information.
     final voice = tts.lang.isEmpty ? 'en-us' : tts.lang;

     // piper may emit empty sentences (e.g., a trailing segment after
     // punctuation); the C API requires every sentence to be non-empty.
     final sentences = piper.PiperPhonemize
         .phonemize(text, voice: voice)
         .where((sentence) => sentence.isNotEmpty)
         .toList();

     final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
       sid: 0,
       speed: 1.0,
       silenceScale: 0.2,
       phonemeCodepoints: sentences,
     );

     // The text has been phonemized already; pass an empty string here.
     final audio = tts.generateWithConfig(text: '', config: genConfig);
     tts.free();

     sherpa_onnx.writeWave(
       filename: './generated-dart.wav',
       samples: audio.samples,
       sampleRate: audio.sampleRate,
     );
   }

.. note::

   In a Flutter app, ``espeak-ng-data`` is a Flutter asset and cannot be used
   as a path directly. On non-web platforms, extract it first:

   .. code-block:: dart

      final dataDir = await piper.extractEspeakNgData();
      piper.PiperPhonemize.initialize(dataDir);

   On web, the data is embedded in the WASM bundle; call
   ``piper.PiperPhonemize.initialize('')`` instead.

Swift
^^^^^

Step 1. Add both packages to your ``Package.swift``:

.. code-block:: swift

   // swift-tools-version:5.9
   import PackageDescription

   let package = Package(
     name: "MyTtsApp",
     platforms: [.iOS(.v13), .macOS(.v10_15)],
     dependencies: [
       .package(url: "https://github.com/k2-fsa/sherpa-onnx", branch: "master"),
       .package(url: "https://github.com/csukuangfj/piper-phonemize", branch: "master"),
     ],
     targets: [
       .executableTarget(
         name: "MyTtsApp",
         dependencies: [
           .product(name: "sherpa-onnx", package: "sherpa-onnx"),
           .product(name: "piper-phonemize", package: "piper-phonemize"),
         ]
       )
     ]
   )

Step 2. Phonemize the text and pass the codepoints to `sherpa-onnx`_:

.. code-block:: swift

   import Foundation
   import SherpaOnnx
   import piper_phonemize

   @main
   struct App {
     static func main() {
       // espeak-ng-data is used only by piper-phonemize; sherpa-onnx itself
       // does not use espeak-ng anymore. The Swift package bundles the data.
       guard let dataDir = piperPhonemizeBundledDataDir() else {
         print("espeak-ng-data not found")
         return
       }
       _ = piperPhonemizeInitialize(dataDir: dataDir)

       let vits = sherpaOnnxOfflineTtsVitsModelConfig(
         model: "./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
         lexicon: "",
         tokens: "./vits-piper-en_US-amy-low/tokens.txt"
       )
       let modelConfig = sherpaOnnxOfflineTtsModelConfig(vits: vits)
       var ttsConfig = sherpaOnnxOfflineTtsConfig(model: modelConfig)
       let tts = SherpaOnnxOfflineTtsWrapper(config: &ttsConfig)

       let text =
         "Today as always, men fall into two groups: slaves and free men."

       // The espeak-ng voice for the phonemizer is taken from the model
       // metadata. Fall back to en-us if the model has no language information.
       let voice = tts.lang.isEmpty ? "en-us" : tts.lang

       guard let result = piperPhonemizeText(text: text, voice: voice) else {
         print("Failed to phonemize the text")
         return
       }
       var phonemeCodepoints: [[Int32]] = []
       for i in 0..<result.numSentences {
         phonemeCodepoints.append(result.getPhonemes(sentenceId: i))
       }

       let genConfig = SherpaOnnxGenerationConfigSwift(
         speed: 1.0, sid: 0, phonemeCodepoints: phonemeCodepoints)

       // The text has been phonemized already; pass an empty string here.
       let audio = tts.generateWithConfig(
         text: "", config: genConfig, callback: nil, arg: nil)
       _ = audio.save(filename: "test-piper-phonemize.wav")
     }
   }

WebAssembly
^^^^^^^^^^^

Step 1. Download the `piper-phonemize`_ wasm bundle and unzip it into
``sherpa-onnx/wasm/tts/piper-phonemize/``:

.. code-block:: bash

   cd sherpa-onnx/wasm/tts
   wget -q https://github.com/csukuangfj/piper-phonemize/releases/download/wasm/piper-phonemize-v1.5.9-wasm-browser.zip
   unzip -q piper-phonemize-v1.5.9-wasm-browser.zip
   mv browser piper-phonemize
   rm piper-phonemize-v1.5.9-wasm-browser.zip

Step 2. Load the scripts in your page in this order. Define ``Module`` before
loading the Emscripten glue script so the glue attaches its runtime to it:

.. code-block:: html

   <script src="piper-phonemize/piper-phonemize.js"></script>
   <script src="app-tts.js"></script>
   <!-- The Emscripten glue must be loaded last, after Module is defined -->
   <script src="piper-phonemize/piper-phonemize-wasm-main.js"></script>

.. code-block:: javascript

   // Define Module before loading piper-phonemize/piper-phonemize-wasm-main.js
   var Module = {
     locateFile: function(path, scriptDirectory = '') {
       // The glue fetches its .data with an empty scriptDirectory; prefix
       // "piper-phonemize/" so the request resolves next to the bundle files.
       return (scriptDirectory || 'piper-phonemize/') + path;
     },
     onRuntimeInitialized: function() {
       piperPhonemizeInitialize();  // initialize espeak-ng
     },
   };

Step 3. Phonemize the text and pass the codepoints to `sherpa-onnx`_, where
``tts`` is created by ``createOfflineTts()`` from ``sherpa-onnx-tts.js``:

.. code-block:: javascript

   const voice = 'en-us';  // espeak-ng voice of your model
   const sentences = piperPhonemize(text, voice);
   // piper may emit empty sentences (e.g., a trailing segment after
   // punctuation); the C API requires every sentence to be non-empty.
   const phonemeCodepoints = sentences.filter((s) => s.length > 0);

   // The text has been phonemized already; pass an empty string here.
   const audio = tts.generateWithConfig('', {
     phonemeCodepoints: phonemeCodepoints,
     sid: 0,
     speed: 1.0,
   });

Notes
-----

- The espeak-ng voice (e.g., ``en-us``) is used only by the phonemizer, not by
  `sherpa-onnx`_. Most examples read it from the model metadata and fall back
  to ``en-us`` if the model has no language information.

- `piper-phonemize`_ may return empty sentences, e.g., a trailing segment
  after punctuation. Please filter them out before passing the result to
  `sherpa-onnx`_.

- If your model uses string tokens, e.g., Chinese pinyin like ``zhong1``, pass
  ``tokens`` instead of ``phoneme_codepoints``; the two are mutually
  exclusive. Please see
  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-matcha-zh-en-phonemize.py>`_

- The ``espeak-ng-data`` directory inside a model archive is used only by the
  external phonemizer; `sherpa-onnx`_ itself does not read it anymore. If you
  use a lexicon, you don't need it at all.

.. _espeak-ng: https://github.com/espeak-ng/espeak-ng
.. _piper-phonemize: https://github.com/csukuangfj/piper-phonemize
