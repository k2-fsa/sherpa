.. _onnx-x-asr-rust-api:

Rust
====

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Rust API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Rust 1.70 or above (the crate downloads pre-built libraries automatically)

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Cargo.toml``:

.. code-block:: toml

   [package]
   name = "x-asr-demo"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   sherpa-onnx = { version = "2.0.1", default-features = false, features = ["static"] }


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main.rs``:

.. code-block:: rust

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in Rust.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   use sherpa_onnx::{
       OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig, Wave,
   };
   use std::time::Instant;

   fn main() {
       let model_dir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03";

       // Step 1: Create the recognizer from the X-ASR model files.
       let mut config = OfflineRecognizerConfig::default();
       config.model_config.transducer = OfflineTransducerModelConfig {
           encoder: Some(format!("{model_dir}/encoder-epoch-99-avg-1.int8.onnx")),
           decoder: Some(format!("{model_dir}/decoder-epoch-99-avg-1.onnx")),
           joiner: Some(format!("{model_dir}/joiner-epoch-99-avg-1.int8.onnx")),
       };
       config.model_config.tokens = Some(format!("{model_dir}/tokens.txt"));
       config.model_config.num_threads = 2;
       config.model_config.debug = false;

       let recognizer =
           OfflineRecognizer::create(&config).expect("Failed to create OfflineRecognizer");

       // Step 2: Read the wave file you want to transcribe.
       let filename = format!("{model_dir}/test_wavs/0.wav");
       let wave = Wave::read(&filename).expect("Failed to read the wave file");

       // Step 3: Create a stream, send the whole audio to it, and decode.
       let start = Instant::now();
       let stream = recognizer.create_stream();
       stream.accept_waveform(wave.sample_rate(), wave.samples());
       recognizer.decode(&stream);

       // Step 4: Print the result. X-ASR punct models also add punctuations.
       println!("{}", filename);
       let elapsed = start.elapsed().as_secs_f64();
       println!("{}", stream.get_result().unwrap().text);

       let audio_duration = wave.samples().len() as f64 / wave.sample_rate() as f64;
       println!("audio duration: {:.3} s", audio_duration);
       println!("elapsed seconds: {:.3} s", elapsed);
       println!(
           "RTF = {:.3}/{:.3} = {:.3}",
           elapsed, audio_duration, elapsed / audio_duration
       );
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Cargo.toml
   ├── sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   │   ├── bpe.model
   │   ├── decoder-epoch-99-avg-1.onnx
   │   ├── encoder-epoch-99-avg-1.int8.onnx
   │   ├── joiner-epoch-99-avg-1.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── src
       └── main.rs

   4 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   166B 19 Nov 12:23 Cargo.toml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cargo run --release

Output
^^^^^^

.. code-block:: text

       Updating crates.io index
        Locking 146 packages to latest compatible versions
    Downloading crates ...
     Downloaded zip v9.7.2
      Compiling proc-macro2 v1.0.107
      Compiling quote v1.0.47
      Compiling unicode-ident v1.0.27
      Compiling libc v0.2.191
      Compiling shlex v2.0.1
      Compiling find-msvc-tools v0.1.15
      Compiling cfg-if v1.0.5
      Compiling stable_deref_trait v1.2.1
      Compiling typenum v1.20.1
      Compiling pkg-config v0.3.34
      Compiling writeable v0.6.4
      Compiling litemap v0.8.3
      Compiling utf_types v1.0.0
      Compiling cmov v0.5.4
      Compiling const-oid v0.10.3
      Compiling icu_properties_data v2.4.0
      Compiling icu_normalizer_data v2.4.0
      Compiling rustversion v1.0.23
      Compiling utf8_iter v1.1.0
      Compiling ctutils v0.4.8
      Compiling crc32fast v1.5.5
      Compiling simd-adler32 v0.3.10
      Compiling smallvec v1.16.6
      Compiling bitflags v2.13.2
      Compiling zstd-safe v8.3.0
      Compiling log v0.4.35
      Compiling rustix v1.1.6
      Compiling hybrid-array v0.4.15
      Compiling untrusted v0.9.0
      Compiling adler2 v2.0.1
      Compiling cfg_aliases v0.2.2
      Compiling miniz_oxide v0.9.1
      Compiling percent-encoding v2.3.2
      Compiling nix v0.32.0
      Compiling rustls v0.23.46
      Compiling zlib-rs v0.6.8
      Compiling getrandom v0.4.3
      Compiling crypto-common v0.2.2
      Compiling inout v0.2.6
      Compiling filetime v0.2.30
      Compiling cipher v0.5.2
      Compiling form_urlencoded v1.2.2
      Compiling powerfmt v0.2.1
      Compiling hashbrown v0.17.1
      Compiling num-conv v0.2.2
      Compiling once_cell v1.21.4
      Compiling cpubits v0.1.1
      Compiling equivalent v1.0.3
      Compiling libbz2-rs-sys v0.2.6
      Compiling subtle v2.6.1
      Compiling deranged v0.5.9
      Compiling flate2 v1.1.11
      Compiling time-core v0.1.9
      Compiling bumpalo v3.20.4
      Compiling time v0.3.55
      Compiling zopfli v0.8.4
      Compiling bzip2 v0.6.1
      Compiling indexmap v2.14.2
      Compiling constant_time_eq v0.6.1
      Compiling memchr v2.8.3
      Compiling serde_core v1.0.229
      Compiling base64 v0.22.1
      Compiling typed-path v0.13.0
      Compiling deflate64 v0.1.12
      Compiling ppmd-rust v1.5.2
      Compiling zmij v1.0.23
      Compiling serde v1.0.229
      Compiling jobserver v0.1.35
      Compiling cpufeatures v0.3.1
      Compiling getrandom v0.2.17
      Compiling errno v0.3.15
      Compiling cc v1.11.2
      Compiling serde_json v1.0.154
      Compiling itoa v1.0.19
      Compiling syn v3.0.10
      Compiling syn v2.0.119
      Compiling zstd-sys v2.1.3+zstd.1.5.7
      Compiling ring v0.17.14
      Compiling bzip2-sys v0.1.13+1.0.8
      Compiling xattr v1.7.0
      Compiling tar v0.4.46
      Compiling synstructure v0.14.0
      Compiling zerovec-derive v0.11.7
      Compiling displaydoc v0.2.7
      Compiling zerofrom-derive v0.1.8
      Compiling yoke-derive v0.8.4
      Compiling zerofrom v0.1.8
      Compiling serde_derive v1.0.229
      Compiling zeroize_derive v1.5.0
      Compiling zeroize v1.9.2
      Compiling bzip2 v0.4.4
      Compiling yoke v0.8.4
      Compiling block-buffer v0.12.1
      Compiling rustls-pki-types v1.16.0
      Compiling aes v0.9.4
      Compiling digest v0.11.3
      Compiling sha2 v0.11.1
      Compiling hmac v0.13.0
      Compiling sha1 v0.11.0
      Compiling rustls-webpki v0.103.17
      Compiling webpki-roots v1.0.12
      Compiling pbkdf2 v0.13.0
      Compiling webpki-roots v0.26.11
      Compiling lzma-rust2 v0.25.0
      Compiling zstd v0.14.4
      Compiling zip v9.7.2
      Compiling zerovec v0.11.9
      Compiling zerotrie v0.2.5
      Compiling tinystr v0.8.5
      Compiling potential_utf v0.1.6
      Compiling icu_collections v2.4.0
      Compiling icu_locale_core v2.4.0
      Compiling icu_provider v2.4.0
      Compiling icu_normalizer v2.4.0
      Compiling icu_properties v2.4.0
      Compiling idna_adapter v1.2.2
      Compiling idna v1.1.0
      Compiling url v2.5.8
      Compiling ureq v2.12.1
      Compiling sherpa-onnx-sys v2.0.1
      Compiling sherpa-onnx v2.0.1
      Compiling x-asr-demo v0.1.0 (/private/tmp/java-doc-work/layouts/rust-offline)
       Finished `release` profile [optimized] target(s) in 26.02s
        Running `target/release/x-asr-demo`
   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.196 s
   RTF = 0.196/10.053 = 0.019

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Streaming X-ASR
---------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Cargo.toml``:

.. code-block:: toml

   [package]
   name = "x-asr-demo"
   version = "0.1.0"
   edition = "2021"

   [dependencies]
   sherpa-onnx = { version = "2.0.1", default-features = false, features = ["static"] }


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``src/main.rs``:

.. code-block:: rust

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in Rust.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   use sherpa_onnx::{OnlineRecognizer, OnlineRecognizerConfig, Wave};
   use std::time::Instant;

   fn main() {
       let model_dir =
           "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05";

       // Step 1: Create the recognizer from the X-ASR model files.
       let mut config = OnlineRecognizerConfig::default();
       config.model_config.transducer.encoder = Some(format!("{model_dir}/encoder.int8.onnx"));
       config.model_config.transducer.decoder = Some(format!("{model_dir}/decoder.onnx"));
       config.model_config.transducer.joiner = Some(format!("{model_dir}/joiner.int8.onnx"));
       config.model_config.tokens = Some(format!("{model_dir}/tokens.txt"));
       config.model_config.num_threads = 2;
       config.model_config.debug = false;
       config.decoding_method = Some("greedy_search".to_string());

       let recognizer = OnlineRecognizer::create(&config).expect("Failed to create OnlineRecognizer");

       // Step 2: Read the wave file you want to transcribe.
       let filename = format!("{model_dir}/test_wavs/0.wav");
       let wave = Wave::read(&filename).expect("Failed to read the wave file");

       // Step 3: Create a stream and feed the audio in small chunks, just like
       // a microphone would do in a real-time application.
       let start = Instant::now();
       let stream = recognizer.create_stream();

       // 0.2 second for 16 kHz audio
       let chunk_size = 3200;

       let mut last = String::new();
       for chunk in wave.samples().chunks(chunk_size) {
           stream.accept_waveform(wave.sample_rate(), chunk);

           while recognizer.is_ready(&stream) {
               recognizer.decode(&stream);
           }

           // Print the latest partial result when it changes.
           if let Some(result) = recognizer.get_result(&stream) {
               if !result.text.is_empty() && result.text != last {
                   last = result.text.clone();
                   println!("{}", result.text);
               }
           }
       }

       // Send some tail padding so the last words are flushed out.
       let tail = vec![0.0f32; (0.5 * wave.sample_rate() as f32) as usize];
       stream.accept_waveform(wave.sample_rate(), &tail);
       while recognizer.is_ready(&stream) {
           recognizer.decode(&stream);
       }

       // Step 4: Print the final result.
       let result = recognizer.get_result(&stream).unwrap();
       let elapsed = start.elapsed().as_secs_f64();
       println!("Final result: {}", result.text);

       let audio_duration = wave.samples().len() as f64 / wave.sample_rate() as f64;
       println!("audio duration: {:.3} s", audio_duration);
       println!("elapsed seconds: {:.3} s", elapsed);
       println!(
           "RTF = {:.3}/{:.3} = {:.3}",
           elapsed, audio_duration, elapsed / audio_duration
       );
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Cargo.toml
   ├── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   │   ├── bpe.model
   │   ├── decoder.onnx
   │   ├── encoder.int8.onnx
   │   ├── joiner.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── src
       └── main.rs

   4 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   166B 19 Nov 12:23 Cargo.toml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 src

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cargo run --release

Output
^^^^^^

.. code-block:: text

       Updating crates.io index
        Locking 146 packages to latest compatible versions
      Compiling proc-macro2 v1.0.107
      Compiling quote v1.0.47
      Compiling unicode-ident v1.0.27
      Compiling libc v0.2.191
      Compiling find-msvc-tools v0.1.15
      Compiling shlex v2.0.1
      Compiling cfg-if v1.0.5
      Compiling stable_deref_trait v1.2.1
      Compiling typenum v1.20.1
      Compiling pkg-config v0.3.34
      Compiling writeable v0.6.4
      Compiling utf_types v1.0.0
      Compiling litemap v0.8.3
      Compiling cmov v0.5.4
      Compiling icu_properties_data v2.4.0
      Compiling utf8_iter v1.1.0
      Compiling icu_normalizer_data v2.4.0
      Compiling const-oid v0.10.3
      Compiling crc32fast v1.5.5
      Compiling rustversion v1.0.23
      Compiling ctutils v0.4.8
      Compiling simd-adler32 v0.3.10
      Compiling smallvec v1.16.6
      Compiling bitflags v2.13.2
      Compiling untrusted v0.9.0
      Compiling rustix v1.1.6
      Compiling cfg_aliases v0.2.2
      Compiling zstd-safe v8.3.0
      Compiling log v0.4.35
      Compiling adler2 v2.0.1
      Compiling nix v0.32.0
      Compiling hybrid-array v0.4.15
      Compiling miniz_oxide v0.9.1
      Compiling filetime v0.2.30
      Compiling percent-encoding v2.3.2
      Compiling zlib-rs v0.6.8
      Compiling rustls v0.23.46
      Compiling getrandom v0.4.3
      Compiling crypto-common v0.2.2
      Compiling inout v0.2.6
      Compiling form_urlencoded v1.2.2
      Compiling deranged v0.5.9
      Compiling syn v3.0.10
      Compiling syn v2.0.119
      Compiling cipher v0.5.2
      Compiling cpubits v0.1.1
      Compiling hashbrown v0.17.1
      Compiling equivalent v1.0.3
      Compiling bumpalo v3.20.4
      Compiling num-conv v0.2.2
      Compiling powerfmt v0.2.1
      Compiling time-core v0.1.9
      Compiling once_cell v1.21.4
      Compiling jobserver v0.1.35
      Compiling cpufeatures v0.3.1
      Compiling getrandom v0.2.17
      Compiling errno v0.3.15
      Compiling libbz2-rs-sys v0.2.6
      Compiling constant_time_eq v0.6.1
      Compiling subtle v2.6.1
      Compiling time v0.3.55
      Compiling zopfli v0.8.4
      Compiling cc v1.11.2
      Compiling indexmap v2.14.2
      Compiling ppmd-rust v1.5.2
      Compiling bzip2 v0.6.1
      Compiling deflate64 v0.1.12
      Compiling typed-path v0.13.0
      Compiling serde_core v1.0.229
      Compiling base64 v0.22.1
      Compiling memchr v2.8.3
      Compiling zmij v1.0.23
      Compiling synstructure v0.14.0
      Compiling flate2 v1.1.11
      Compiling xattr v1.7.0
      Compiling serde_json v1.0.154
      Compiling tar v0.4.46
      Compiling serde v1.0.229
      Compiling itoa v1.0.19
      Compiling ring v0.17.14
      Compiling zerofrom-derive v0.1.8
      Compiling yoke-derive v0.8.4
      Compiling zerovec-derive v0.11.7
      Compiling displaydoc v0.2.7
      Compiling zstd-sys v2.1.3+zstd.1.5.7
      Compiling bzip2-sys v0.1.13+1.0.8
      Compiling serde_derive v1.0.229
      Compiling zeroize_derive v1.5.0
      Compiling zerofrom v0.1.8
      Compiling zeroize v1.9.2
      Compiling bzip2 v0.4.4
      Compiling yoke v0.8.4
      Compiling block-buffer v0.12.1
      Compiling rustls-pki-types v1.16.0
      Compiling aes v0.9.4
      Compiling digest v0.11.3
      Compiling sha2 v0.11.1
      Compiling rustls-webpki v0.103.17
      Compiling webpki-roots v1.0.12
      Compiling hmac v0.13.0
      Compiling sha1 v0.11.0
      Compiling webpki-roots v0.26.11
      Compiling pbkdf2 v0.13.0
      Compiling zerovec v0.11.9
      Compiling zerotrie v0.2.5
      Compiling lzma-rust2 v0.25.0
      Compiling zstd v0.14.4
      Compiling zip v9.7.2
      Compiling tinystr v0.8.5
      Compiling potential_utf v0.1.6
      Compiling icu_collections v2.4.0
      Compiling icu_locale_core v2.4.0
      Compiling icu_provider v2.4.0
      Compiling icu_normalizer v2.4.0
      Compiling icu_properties v2.4.0
      Compiling idna_adapter v1.2.2
      Compiling idna v1.1.0
      Compiling url v2.5.8
      Compiling ureq v2.12.1
      Compiling sherpa-onnx-sys v2.0.1
      Compiling sherpa-onnx v2.0.1
      Compiling x-asr-demo v0.1.0 (/private/tmp/java-doc-work/layouts/rust-streaming)
       Finished `release` profile [optimized] target(s) in 19.43s
        Running `target/release/x-asr-demo`
   昨
   昨天是
   昨天是 Mon
   昨天是 Monday
   昨天是 Monday，
   昨天是 Monday， today
   昨天是 Monday， today is 礼
   昨天是 Monday， today is 礼拜二
   昨天是 Monday， today is 礼拜二，
   昨天是 Monday， today is 礼拜二， the day
   昨天是 Monday， today is 礼拜二， the day after
   昨天是 Monday， today is 礼拜二， the day after tomorro
   昨天是 Monday， today is 礼拜二， the day after tomorrow
   昨天是 Monday， today is 礼拜二， the day after tomorrow 是
   Final result: 昨天是 Monday， today is 礼拜二， the day after tomorrow 是星期
   audio duration: 10.053 s
   elapsed seconds: 0.355 s
   RTF = 0.355/10.053 = 0.035

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

