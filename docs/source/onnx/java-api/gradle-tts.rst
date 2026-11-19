.. _java-gradle-tts:

Gradle: TTS
===========

This example shows how to use a VITS model from `piper <https://github.com/rhasspy/piper>`_ for text-to-speech. Two text frontends are shown: a lexicon file (no extra dependency) and `piper-phonemize <https://github.com/csukuangfj/piper-phonemize>`_.

Please see `java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_ for more TTS models, e.g., `NonStreamingTtsMatchaEnLexicon.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsMatchaEnLexicon.java>`_, `NonStreamingTtsKokoroEnLexicon.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsKokoroEnLexicon.java>`_, `NonStreamingTtsKittenEnLexicon.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsKittenEnLexicon.java>`_.

It is the Java version of `NonStreamingTtsPiperEnLexicon.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsPiperEnLexicon.java>`_ and `NonStreamingTtsPiperEnPhonemize.java <https://github.com/k2-fsa/sherpa-onnx/blob/master/java-api-examples/NonStreamingTtsPiperEnPhonemize.java>`_ in `sherpa-onnx/java-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/java-api-examples>`_.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- JDK 11 or above
- Gradle 8.x+ (or use the included Gradle wrapper, no separate install needed)

.. warning::

   The sherpa-onnx Java packages are hosted on `JitPack <https://jitpack.io/>`_, so you need to add the JitPack repository.

TTS with a lexicon
------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir sherpa-onnx-java-demo
   cd sherpa-onnx-java-demo

Step 2. Create the build files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``build.gradle``:

.. code-block:: groovy

   plugins {
       id 'application'
       id 'java'
   }

   application {
       mainClass = 'NonStreamingTtsPiperEnLexicon'
   }

   repositories {
       mavenCentral()
       // sherpa-onnx packages are hosted on JitPack
       maven { url 'https://jitpack.io' }
   }

   dependencies {
       implementation 'com.github.k2-fsa:sherpa-onnx:v2.0.1'
   }

   java {
       sourceCompatibility = JavaVersion.VERSION_11
       targetCompatibility = JavaVersion.VERSION_11
   }

Save the following as ``settings.gradle``:

.. code-block:: groovy

   rootProject.name = 'sherpa-onnx-java-example'

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
   tar xf vits-piper-en_GB-cori-medium.tar.bz2
   rm vits-piper-en_GB-cori-medium.tar.bz2

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/NonStreamingTtsPiperEnLexicon.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a piper VITS English TTS model
   // to convert text to speech
   import com.k2fsa.sherpa.onnx.*;

   public class NonStreamingTtsPiperEnLexicon {
     public static void main(String[] args) {
       // please visit
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
       // to download model files
       String model = "./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx";
       String tokens = "./vits-piper-en_GB-cori-medium/tokens.txt";
       String lexicon = "./lexicon-en-us.txt";
       String text =
           "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
               + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
               + " businessman, an official, or a scholar.";

       OfflineTtsVitsModelConfig vitsModelConfig =
           OfflineTtsVitsModelConfig.builder()
               .setModel(model)
               .setTokens(tokens)
               .setLexicon(lexicon)
               .build();

       OfflineTtsModelConfig modelConfig =
           OfflineTtsModelConfig.builder()
               .setVits(vitsModelConfig)
               .setNumThreads(1)
               .setDebug(true)
               .build();

       OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
       OfflineTts tts = new OfflineTts(config);

       int sid = 0;
       float speed = 1.0f;
       GenerationConfig genConfig = new GenerationConfig();
       genConfig.setSid(sid);
       genConfig.setSpeed(speed);
       genConfig.setSilenceScale(config.getSilenceScale());
       long start = System.currentTimeMillis();
       GeneratedAudio audio =
           tts.generateWithConfigAndCallback(text, genConfig, (float[] samples) -> 1);
       long stop = System.currentTimeMillis();

       float timeElapsedSeconds = (stop - start) / 1000.0f;

       float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
       float realTimeFactor = timeElapsedSeconds / audioDuration;

       String waveFilename = "tts-piper-en-lexicon.wav";
       audio.save(waveFilename);
       System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
       System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
       System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
       System.out.printf("-- text: %s\n", text);
       System.out.printf("-- Saved to %s\n", waveFilename);

       tts.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── lexicon-en-us.txt
   ├── settings.gradle
   ├── src
   │   └── main
   │       └── java
   │           └── NonStreamingTtsPiperEnLexicon.java
   └── vits-piper-en_GB-cori-medium
       ├── en_GB-cori-medium.onnx
       ├── en_GB-cori-medium.onnx.json
       ├── espeak-ng-data
       │   ├── af_dict
       │   ├── am_dict
       │   ├── an_dict
       │   ├── ar_dict
       │   └── ... (the rest of the files in espeak-ng-data are not shown here)
       ├── MODEL_CARD
       └── tokens.txt

   42 directories, 363 files

   $ ls -lh
   total 12336
   -rw-r--r--@ 1 fangjun  wheel   421B 19 Nov 14:30 build.gradle
   -rw-r--r--@ 1 fangjun  wheel   6.0M 19 Nov 14:30 lexicon-en-us.txt
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:30 settings.gradle
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:30 src
   drwxr-xr-x@ 7 fangjun  wheel   224B 30 Jun  2025 vits-piper-en_GB-cori-medium

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate the Gradle wrapper (only once; requires Gradle to be installed)
   gradle wrapper

   ./gradlew run

Output
^^^^^^

.. code-block:: text

   > Task :wrapper

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-tts-lexicon/build/reports/problems/problems-report.html

   Deprecated Gradle features were used in this build, making it incompatible with Gradle 10.

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   BUILD SUCCESSFUL in 392ms
   1 actionable task: 1 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html
   > Task :compileJava
   > Task :processResources NO-SOURCE
   > Task :classes

   > Task :run
   WARNING: A restricted method in java.lang.System has been called
   WARNING: java.lang.System::load has been called by com.k2fsa.sherpa.onnx.LibraryUtils in an unnamed module (file:/Users/fangjun/.gradle/caches/modules-2/files-2.1/com.github.k2-fsa.sherpa-onnx/sherpa-onnx-jvm/v2.0.1/f846e563993c1c126d50544cb537200006d1f1bb/sherpa-onnx-jvm-v2.0.1.jar)
   WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
   WARNING: Restricted methods will be blocked in a future release unless native access is enabled

   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/jni/offline-tts.cc:operator():550 config:
   OfflineTtsConfig(model=OfflineTtsModelConfig(vits=OfflineTtsVitsModelConfig(model="./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx", lexicon="./lexicon-en-us.txt", tokens="./vits-piper-en_GB-cori-medium/tokens.txt", noise_scale=0.667, noise_scale_w=0.8, length_scale=1), matcha=OfflineTtsMatchaModelConfig(acoustic_model="", vocoder="", lexicon="", tokens="", data_dir="", noise_scale=1, length_scale=1), kokoro=OfflineTtsKokoroModelConfig(model="", voices="", tokens="", lexicon="", data_dir="", length_scale=1, lang=""), zipvoice=OfflineTtsZipvoiceModelConfig(tokens="", encoder="", decoder="", vocoder="", data_dir="", lexicon="", feat_scale=0.1, t_shift=0.5, target_rms=0.1, guidance_scale=1), kitten=OfflineTtsKittenModelConfig(model="", voices="", tokens="", lexicon="", data_dir="", length_scale=1), pocket=OfflineTtsPocketModelConfig(lm_flow="", lm_main="", encoder="", decoder="", text_conditioner="", vocab_json="", token_scores_json="", voice_embedding_cache_capacity=50), supertonic=OfflineTtsSupertonicModelConfig(duration_predictor="", text_encoder="", vector_estimator="", vocoder="", tts_json="", unicode_indexer="", voice_style=""), num_threads=1, debug=True, provider="cpu"), rule_fsts="", rule_fars="", max_num_sentences=1, silence_scale=0.2)
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-model.cc:Init:163 ---vits model---
   sample_rate=22050
   has_espeak=1
   n_speakers=1
   voice=en
   model_type=vits
   language=English
   comment=piper
   ----------input names----------
   0 input
   1 input_lengths
   2 scales
   ----------output names----------
   0 output


   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:LoadLexicon:623 Loaded lexicon with 168653 entries, max phrase len 11
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:88 GenerationConfig(silence_scale=0.2, speed=1, sid=0, num_steps=5, reference_audio_len=0, reference_sample_rate=0)
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:NormalizeText:252 Raw text: Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'today' -> 'tədˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'as' -> 'æz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'always' -> 'ˈɔːlweɪz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'men' -> 'mˈɛn'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'fall' -> 'fˈɔːl'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'into' -> 'ˌɪntʊ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'two' -> 'tˈuː'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'groups' -> 'ɡɹˈuːps'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:745 Sentence 0: text='Today as always, men fall into two groups:', 50 codepoints, phonemes='tədˈeɪ æz ˈɔːlweɪz, mˈɛn fˈɔːl ˌɪntʊ tˈuː ɡɹˈuːps:'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'slaves' -> 'slˈeɪvz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'and' -> 'ænd'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'free' -> 'fɹˈiː'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'men' -> 'mˈɛn'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:745 Sentence 1: text='slaves and free men.', 23 codepoints, phonemes='slˈeɪvz ænd fɹˈiː mˈɛn.'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'whoever' -> 'huːˈɛvɚ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'does not' -> 'dʌznˈɑːt'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'have' -> 'hˈæv'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:LogMatched:770 Lexicon matched: 'two' -> 'tˈuː'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:LogMatched:770 Lexicon matched: 'thirds' -> 'θˈɜːdz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'of' -> 'ʌv'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'his' -> 'hˈɪz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'day' -> 'dˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'for' -> 'fɔːɹ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'himself' -> 'hɪmsˈɛlf'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'is' -> 'ɪz'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'a' -> 'ˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'slave' -> 'slˈeɪv'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'whatever' -> 'wʌtˈɛvɚ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'he' -> 'hiː'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'may' -> 'mˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'be' -> 'bˈiː'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:745 Sentence 2: text='Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be:', 98 codepoints, phonemes='huːˈɛvɚ dʌznˈɑːt hˈæv tˈuːθˈɜːdz ʌv hˈɪz dˈeɪ fɔːɹ hɪmsˈɛlf, ɪz ˈeɪ slˈeɪv, wʌtˈɛvɚ hiː mˈeɪ bˈiː:'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'a' -> 'ˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'statesman' -> 'stˈeɪtsmən'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'a' -> 'ˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'businessman' -> 'bˈɪznəsmən'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'an' -> 'æn'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'official' -> 'əfˈɪʃəl'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:LookupPieces:825 OOV word skipped: 'or'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'a' -> 'ˈeɪ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:711 Lexicon matched: 'scholar' -> 'skˈɑːlɚ'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:TokenizeFromLexicon:745 Sentence 3: text='a statesman, a businessman, an official, or a scholar.', 56 codepoints, phonemes='ˈeɪ stˈeɪtsmən, ˈeɪ bˈɪznəsmən, æn əfˈɪʃəl, ˈeɪ skˈɑːlɚ.'
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 0: 103 token IDs: [1, 0, 32, 0, 59, 0, 17, 0, 120, 0, 18, 0, 74, 0, 3, 0, 39, 0, 38, 0, 3, 0, 120, 0, 54, 0, 122, 0, 24, 0, 35, 0, 18, 0, 74, 0, 38, 0, 8, 0, 3, 0, 25, 0, 120, 0, 61, 0, 26, 0, 3, 0, 19, 0, 120, 0, 54, 0, 122, 0, 24, 0, 3, 0, 121, 0, 74, 0, 26, 0, 32, 0, 100, 0, 3, 0, 32, 0, 120, 0, 33, 0, 122, 0, 3, 0, 66, 0, 88, 0, 120, 0, 33, 0, 122, 0, 28, 0, 31, 0, 11, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 1: 49 token IDs: [1, 0, 31, 0, 24, 0, 120, 0, 18, 0, 74, 0, 34, 0, 38, 0, 3, 0, 39, 0, 26, 0, 17, 0, 3, 0, 19, 0, 88, 0, 120, 0, 21, 0, 122, 0, 3, 0, 25, 0, 120, 0, 61, 0, 26, 0, 10, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 2: 199 token IDs: [1, 0, 20, 0, 33, 0, 122, 0, 120, 0, 61, 0, 34, 0, 60, 0, 3, 0, 17, 0, 102, 0, 38, 0, 26, 0, 120, 0, 51, 0, 122, 0, 32, 0, 3, 0, 20, 0, 120, 0, 39, 0, 34, 0, 3, 0, 32, 0, 120, 0, 33, 0, 122, 0, 126, 0, 120, 0, 62, 0, 122, 0, 17, 0, 38, 0, 3, 0, 102, 0, 34, 0, 3, 0, 20, 0, 120, 0, 74, 0, 38, 0, 3, 0, 17, 0, 120, 0, 18, 0, 74, 0, 3, 0, 19, 0, 54, 0, 122, 0, 88, 0, 3, 0, 20, 0, 74, 0, 25, 0, 31, 0, 120, 0, 61, 0, 24, 0, 19, 0, 8, 0, 3, 0, 74, 0, 38, 0, 3, 0, 120, 0, 18, 0, 74, 0, 3, 0, 31, 0, 24, 0, 120, 0, 18, 0, 74, 0, 34, 0, 8, 0, 3, 0, 35, 0, 102, 0, 32, 0, 120, 0, 61, 0, 34, 0, 60, 0, 3, 0, 20, 0, 21, 0, 122, 0, 3, 0, 25, 0, 120, 0, 18, 0, 74, 0, 3, 0, 15, 0, 120, 0, 21, 0, 122, 0, 11, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 3: 115 token IDs: [1, 0, 120, 0, 18, 0, 74, 0, 3, 0, 31, 0, 32, 0, 120, 0, 18, 0, 74, 0, 32, 0, 31, 0, 25, 0, 59, 0, 26, 0, 8, 0, 3, 0, 120, 0, 18, 0, 74, 0, 3, 0, 15, 0, 120, 0, 74, 0, 38, 0, 26, 0, 59, 0, 31, 0, 25, 0, 59, 0, 26, 0, 8, 0, 3, 0, 39, 0, 26, 0, 3, 0, 59, 0, 19, 0, 120, 0, 74, 0, 96, 0, 59, 0, 24, 0, 8, 0, 3, 0, 120, 0, 18, 0, 74, 0, 3, 0, 31, 0, 23, 0, 120, 0, 51, 0, 122, 0, 24, 0, 60, 0, 10, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:GenerateInBatches:362 Text is too long. Split it into 4 batches. batch size: 1. Number of sentences: 4
   -- elapsed : 0.790 seconds
   -- audio duration: 12.576 seconds
   -- real-time factor (RTF): 0.063
   -- text: Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.
   -- Saved to tts-piper-en-lexicon.wav

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-tts-lexicon/build/reports/problems/problems-report.html

   Deprecated Gradle features were used in this build, making it incompatible with Gradle 10.

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   BUILD SUCCESSFUL in 2s
   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

.. note::

   The generated audio is saved to ``tts-piper-en-lexicon.wav``.

TTS with piper-phonemize
------------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir sherpa-onnx-java-demo
   cd sherpa-onnx-java-demo

Step 2. Create the build files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``build.gradle``:

.. code-block:: groovy

   plugins {
       id 'application'
       id 'java'
   }

   application {
       mainClass = 'NonStreamingTtsPiperEnPhonemize'
   }

   repositories {
       mavenCentral()
       // sherpa-onnx packages are hosted on JitPack
       maven { url 'https://jitpack.io' }
   }

   dependencies {
       implementation 'com.github.k2-fsa:sherpa-onnx:v2.0.1'

       // piper-phonemize: JVM core API
       implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-jvm:v1.5.9'

       // piper-phonemize: native lib - uncomment ONE for your platform
       implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-osx-aarch64:v1.5.9'
       // implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-osx-x64:v1.5.9'
       // implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-linux-x64:v1.5.9'
       // implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-linux-aarch64:v1.5.9'
       // implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-win-x64:v1.5.9'
       // implementation 'com.github.csukuangfj.piper-phonemize:piper-phonemize-native-lib-win-arm64:v1.5.9'
   }

   java {
       sourceCompatibility = JavaVersion.VERSION_11
       targetCompatibility = JavaVersion.VERSION_11
   }

Save the following as ``settings.gradle``:

.. code-block:: groovy

   rootProject.name = 'sherpa-onnx-java-example'

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
   tar xf vits-piper-en_GB-cori-medium.tar.bz2
   rm vits-piper-en_GB-cori-medium.tar.bz2

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main/java/NonStreamingTtsPiperEnPhonemize.java``:

.. code-block:: java

   // Copyright 2024 Xiaomi Corporation

   // This file shows how to use a piper VITS English TTS model
   // to convert text to speech.
   //
   // The text is phonemized outside of sherpa-onnx with piper-phonemize
   // (https://github.com/csukuangfj/piper-phonemize) and the resulting phoneme
   // codepoints are passed via GenerationConfig.phonemeCodepoints. No lexicon
   // file and no espeak-ng-data are needed.
   //
   // If you want to use a lexicon instead, please see
   // NonStreamingTtsPiperEnLexicon.java.
   import com.github.csukuangfj.piper.phonemize.PiperPhonemize;
   import com.k2fsa.sherpa.onnx.*;

   public class NonStreamingTtsPiperEnPhonemize {
     public static void main(String[] args) {
       // please visit
       // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
       // to download model files
       String model = "./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx";
       String tokens = "./vits-piper-en_GB-cori-medium/tokens.txt";
       String text =
           "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
               + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
               + " businessman, an official, or a scholar.";

       OfflineTtsVitsModelConfig vitsModelConfig =
           OfflineTtsVitsModelConfig.builder()
               .setModel(model)
               .setTokens(tokens)
               .build();

       OfflineTtsModelConfig modelConfig =
           OfflineTtsModelConfig.builder()
               .setVits(vitsModelConfig)
               .setNumThreads(1)
               .setDebug(true)
               .build();

       OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
       OfflineTts tts = new OfflineTts(config);

       System.out.printf("piper-phonemize version: %s\n", PiperPhonemize.getVersionStr());
       // initialize() extracts the espeak-ng-data embedded in piper-phonemize.
       // That data only feeds the piper phonemizer; sherpa-onnx itself does not
       // use espeak-ng anymore.
       PiperPhonemize.initialize();

       // The espeak-ng voice for the phonemizer is taken from the model metadata.
       // Fall back to en-us if the model has no language information.
       String voice = tts.getLang();
       if (voice == null || voice.isEmpty()) {
         System.out.println("Warning: the model has no language information. Falling back to en-us.");
         voice = "en-us";
       }
       System.out.printf("Phonemizer language: %s\n", voice);

       int[][] phonemeCodepoints;
       try (PiperPhonemize.Result result = PiperPhonemize.text(text, voice)) {
         if (result == null) {
           throw new RuntimeException("Failed to phonemize the text");
         }
         phonemeCodepoints = new int[result.getNumSentences()][];
         for (int i = 0; i < result.getNumSentences(); i++) {
           phonemeCodepoints[i] = result.getPhonemes(i);
         }
       }
       System.out.printf("Number of phonemized sentences: %d\n", phonemeCodepoints.length);

       int sid = 0;
       float speed = 1.0f;
       GenerationConfig genConfig = new GenerationConfig();
       genConfig.setSid(sid);
       genConfig.setSpeed(speed);
       genConfig.setSilenceScale(config.getSilenceScale());
       genConfig.setPhonemeCodepoints(phonemeCodepoints);

       long start = System.currentTimeMillis();
       // The text has already been phonemized; pass an empty string here.
       GeneratedAudio audio = tts.generateWithConfig("", genConfig);
       long stop = System.currentTimeMillis();

       float timeElapsedSeconds = (stop - start) / 1000.0f;

       float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
       float realTimeFactor = timeElapsedSeconds / audioDuration;

       String waveFilename = "tts-piper-en-phonemize.wav";
       audio.save(waveFilename);
       System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
       System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
       System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
       System.out.printf("-- text: %s\n", text);
       System.out.printf("-- Saved to %s\n", waveFilename);

       tts.release();
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── build.gradle
   ├── settings.gradle
   ├── src
   │   └── main
   │       └── java
   │           └── NonStreamingTtsPiperEnPhonemize.java
   └── vits-piper-en_GB-cori-medium
       ├── en_GB-cori-medium.onnx
       ├── en_GB-cori-medium.onnx.json
       ├── espeak-ng-data
       │   ├── af_dict
       │   ├── am_dict
       │   ├── an_dict
       │   ├── ar_dict
       │   └── ... (the rest of the files in espeak-ng-data are not shown here)
       ├── MODEL_CARD
       └── tokens.txt

   42 directories, 362 files

   $ ls -lh
   total 16
   -rw-r--r--@ 1 fangjun  wheel   1.2K 19 Nov 14:32 build.gradle
   -rw-r--r--@ 1 fangjun  wheel    45B 19 Nov 14:32 settings.gradle
   drwxr-xr-x@ 3 fangjun  wheel    96B 19 Nov 14:32 src
   drwxr-xr-x@ 7 fangjun  wheel   224B 30 Jun  2025 vits-piper-en_GB-cori-medium

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate the Gradle wrapper (only once; requires Gradle to be installed)
   gradle wrapper

   ./gradlew run

Output
^^^^^^

.. code-block:: text

   > Task :wrapper UP-TO-DATE

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-tts-phonemize/build/reports/problems/problems-report.html

   Deprecated Gradle features were used in this build, making it incompatible with Gradle 10.

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   BUILD SUCCESSFUL in 238ms
   1 actionable task: 1 up-to-date
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html
   > Task :compileJava
   > Task :processResources NO-SOURCE
   > Task :classes

   > Task :run
   WARNING: A restricted method in java.lang.System has been called
   WARNING: java.lang.System::load has been called by com.k2fsa.sherpa.onnx.LibraryUtils in an unnamed module (file:/Users/fangjun/.gradle/caches/modules-2/files-2.1/com.github.k2-fsa.sherpa-onnx/sherpa-onnx-jvm/v2.0.1/f846e563993c1c126d50544cb537200006d1f1bb/sherpa-onnx-jvm-v2.0.1.jar)
   WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
   WARNING: Restricted methods will be blocked in a future release unless native access is enabled

   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/jni/offline-tts.cc:operator():550 config:
   OfflineTtsConfig(model=OfflineTtsModelConfig(vits=OfflineTtsVitsModelConfig(model="./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx", lexicon="", tokens="./vits-piper-en_GB-cori-medium/tokens.txt", noise_scale=0.667, noise_scale_w=0.8, length_scale=1), matcha=OfflineTtsMatchaModelConfig(acoustic_model="", vocoder="", lexicon="", tokens="", data_dir="", noise_scale=1, length_scale=1), kokoro=OfflineTtsKokoroModelConfig(model="", voices="", tokens="", lexicon="", data_dir="", length_scale=1, lang=""), zipvoice=OfflineTtsZipvoiceModelConfig(tokens="", encoder="", decoder="", vocoder="", data_dir="", lexicon="", feat_scale=0.1, t_shift=0.5, target_rms=0.1, guidance_scale=1), kitten=OfflineTtsKittenModelConfig(model="", voices="", tokens="", lexicon="", data_dir="", length_scale=1), pocket=OfflineTtsPocketModelConfig(lm_flow="", lm_main="", encoder="", decoder="", text_conditioner="", vocab_json="", token_scores_json="", voice_embedding_cache_capacity=50), supertonic=OfflineTtsSupertonicModelConfig(duration_predictor="", text_encoder="", vector_estimator="", vocoder="", tts_json="", unicode_indexer="", voice_style=""), num_threads=1, debug=True, provider="cpu"), rule_fsts="", rule_fars="", max_num_sentences=1, silence_scale=0.2)
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-model.cc:Init:163 ---vits model---
   sample_rate=22050
   has_espeak=1
   n_speakers=1
   voice=en
   model_type=vits
   language=English
   comment=piper
   ----------input names----------
   0 input
   1 input_lengths
   2 scales
   ----------output names----------
   0 output


   piper-phonemize version: 1.5.9
   Phonemizer language: en
   Number of phonemized sentences: 2
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:88 GenerationConfig(silence_scale=0.2, speed=1, sid=0, num_steps=5, reference_audio_len=0, reference_sample_rate=0, phoneme_codepoints=[2 sentences])
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 0: 151 token IDs: [1, 0, 32, 0, 59, 0, 17, 0, 120, 0, 18, 0, 74, 0, 3, 0, 14, 0, 38, 0, 3, 0, 120, 0, 54, 0, 122, 0, 24, 0, 35, 0, 18, 0, 74, 0, 38, 0, 8, 0, 3, 0, 25, 0, 120, 0, 61, 0, 26, 0, 3, 0, 19, 0, 120, 0, 54, 0, 122, 0, 24, 0, 3, 0, 121, 0, 74, 0, 26, 0, 32, 0, 100, 0, 3, 0, 32, 0, 120, 0, 33, 0, 122, 0, 3, 0, 66, 0, 88, 0, 120, 0, 33, 0, 122, 0, 28, 0, 31, 0, 11, 0, 3, 0, 31, 0, 24, 0, 120, 0, 18, 0, 74, 0, 34, 0, 38, 0, 3, 0, 14, 0, 26, 0, 17, 0, 3, 0, 19, 0, 88, 0, 120, 0, 21, 0, 122, 0, 3, 0, 25, 0, 120, 0, 61, 0, 26, 0, 10, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:Generate:153 sentence 1: 295 token IDs: [1, 0, 20, 0, 33, 0, 122, 0, 120, 0, 61, 0, 34, 0, 59, 0, 3, 0, 17, 0, 102, 0, 38, 0, 26, 0, 121, 0, 52, 0, 32, 0, 3, 0, 20, 0, 14, 0, 34, 0, 3, 0, 32, 0, 120, 0, 33, 0, 122, 0, 126, 0, 120, 0, 62, 0, 122, 0, 17, 0, 38, 0, 3, 0, 52, 0, 34, 0, 3, 0, 20, 0, 74, 0, 38, 0, 3, 0, 17, 0, 120, 0, 18, 0, 74, 0, 3, 0, 19, 0, 54, 0, 122, 0, 3, 0, 20, 0, 74, 0, 25, 0, 31, 0, 120, 0, 61, 0, 24, 0, 19, 0, 8, 0, 3, 0, 74, 0, 38, 0, 3, 0, 50, 0, 3, 0, 31, 0, 24, 0, 120, 0, 18, 0, 74, 0, 34, 0, 8, 0, 3, 0, 35, 0, 52, 0, 32, 0, 120, 0, 61, 0, 34, 0, 59, 0, 3, 0, 20, 0, 21, 0, 122, 0, 3, 0, 25, 0, 120, 0, 18, 0, 74, 0, 3, 0, 15, 0, 120, 0, 21, 0, 122, 0, 11, 0, 3, 0, 50, 0, 3, 0, 31, 0, 32, 0, 120, 0, 18, 0, 74, 0, 32, 0, 31, 0, 25, 0, 59, 0, 26, 0, 8, 0, 3, 0, 50, 0, 3, 0, 15, 0, 120, 0, 74, 0, 38, 0, 26, 0, 59, 0, 31, 0, 25, 0, 59, 0, 26, 0, 8, 0, 3, 0, 50, 0, 26, 0, 3, 0, 59, 0, 19, 0, 120, 0, 74, 0, 96, 0, 59, 0, 24, 0, 8, 0, 3, 0, 54, 0, 122, 0, 88, 0, 3, 0, 50, 0, 3, 0, 31, 0, 23, 0, 120, 0, 52, 0, 24, 0, 59, 0, 10, 0, 2]
   /Users/runner/work/sherpa-onnx/sherpa-onnx/sherpa-onnx/csrc/offline-tts-vits-impl.h:GenerateInBatches:362 Text is too long. Split it into 2 batches. batch size: 1. Number of sentences: 2
   -- elapsed : 0.773 seconds
   -- audio duration: 11.570 seconds
   -- real-time factor (RTF): 0.067
   -- text: Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.
   -- Saved to tts-piper-en-phonemize.wav

   [Incubating] Problems report is available at: file:///private/tmp/java-doc-work/layouts-java/gradle-tts-phonemize/build/reports/problems/problems-report.html

   Deprecated Gradle features were used in this build, making it incompatible with Gradle 10.

   You can use '--warning-mode all' to show the individual deprecation warnings and determine if they come from your own scripts or plugins.

   For more on this, please refer to https://docs.gradle.org/9.6.1/userguide/command_line_interface.html#sec:command_line_warnings in the Gradle documentation.

   BUILD SUCCESSFUL in 3s
   2 actionable tasks: 2 executed
   Consider enabling configuration cache to speed up this build: https://docs.gradle.org/9.6.1/userguide/configuration_cache_enabling.html

.. note::

   The generated audio is saved to ``tts-piper-en-phonemize.wav``. The piper-phonemize packages are hosted on JitPack like the sherpa-onnx packages. Note that piper-phonemize has no all-in-one package: the JVM API and the native libraries are separate artifacts, so both dependencies above are needed. The native library artifact depends on your platform, see the comments in the build file above.

