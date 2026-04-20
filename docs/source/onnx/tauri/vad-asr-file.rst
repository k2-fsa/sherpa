.. _vad-asr-file:

Non-Streaming Speech Recognition from File
==========================================

This app is a `Tauri v2 <https://v2.tauri.app/>`_ desktop application that
performs **non-streaming speech recognition on audio and video files**.
It uses `Silero VAD <https://github.com/snakers4/silero-vad>`_ to detect
speech segments and then runs an offline ASR model on each segment.

.. note::

   This app works with **files** (e.g., ``.wav``, ``.mp3``, ``.mp4``),
   not with microphone input. You can use it to transcribe audio/video
   files and **generate subtitles** in SRT format.

Features
--------

- 62 ASR models supported (SenseVoice, Paraformer, Whisper, Transducer,
  Moonshine, Parakeet, Canary, Qwen3 ASR, etc.)
- Audio/video playback with waveform display
- SRT subtitle export
- Segment WAV export — save individual speech segments as WAV files
- Progress tracking with cancellation support
- Cross-platform — macOS (universal), Linux (x64/aarch64), Windows (x64)
- Pure-Rust audio decoding via `symphonia <https://github.com/pdeljanov/Symphonia>`_ —
  no system dependencies

Supported Audio Formats
-----------------------

Any format supported by symphonia:

- **Audio**: MP3, FLAC, AAC, OGG/Vorbis, WAV, AIFF, ADPCM
- **Video**: MP4/M4A, MKV, WebM (audio track extracted)

Pre-built Apps
--------------

Pre-built apps for all 62 models and 4 platforms are available at:

  `<https://huggingface.co/csukuangfj2/tauri-app>`_

The filename follows the pattern:

  ``vad-asr-{version}-{lang}-{model}-{platform}.{ext}``

where:

- ``version``: the current version, e.g., |sherpa_onnx_version|
- ``lang``: the language of the model, e.g., ``en`` for English, ``zh`` for Chinese
- ``model``: the name of the model used in the app
- ``platform``: one of ``linux-x64``, ``linux-aarch64``, ``universal.app`` (macOS), ``windows-x64``
- ``ext``: ``.tar.gz`` for Linux, ``.tar.bz2`` for macOS, ``.zip`` for Windows

You can download all supported models from
`<https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>`_.

Building the App
----------------

Prerequisites
~~~~~~~~~~~~~

- `Rust <https://www.rust-lang.org/tools/install>`_ (stable)
- `Tauri CLI <https://v2.tauri.app/start/prerequisites/>`_:

.. code-block:: bash

   cargo install tauri-cli

Example 1: SenseVoice (multilingual, Chinese/English/Japanese/Korean/Cantonese)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses `model type 15 <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/model_registry.rs>`_ — ``sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17``.
It supports Chinese, English, Japanese, Korean, and Cantonese.

Step 1: Download the model and Silero VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd tauri-examples/non-streaming-speech-recognition-from-file

   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
   tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
   rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

Step 2: Copy assets
^^^^^^^^^^^^^^^^^^^

Tauri bundles files from ``src-tauri/assets/``. Place the model directory
and ``silero_vad.onnx`` inside it:

.. code-block:: bash

   mkdir -p src-tauri/assets
   cp -a sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17 src-tauri/assets/
   cp -a silero_vad.onnx src-tauri/assets/

After copying, you can delete the original model directory to save disk space:

.. code-block:: bash

   rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17

This gives:

.. code-block:: text

   src-tauri/assets/
   ├── silero_vad.onnx
   └── sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
       ├── model.int8.onnx
       └── tokens.txt

.. tip::

   SenseVoice supports a homophone replacer. To enable it, also download
   ``lexicon.txt`` and ``replace.fst`` into ``src-tauri/assets/``:

   .. code-block:: bash

      curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt
      curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
      cp -a lexicon.txt replace.fst src-tauri/assets/

   The app works fine without these files.

Step 3: Set model type
^^^^^^^^^^^^^^^^^^^^^^

Edit `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/lib.rs>`_ and make sure the constants match:

.. code-block:: rust

   const MODEL_TYPE: u32 = 15;
   const MODEL_NAME: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";

Step 4: Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cargo tauri dev

To build a release binary:

.. code-block:: bash

   cargo tauri build

The output is in ``src-tauri/target/release/bundle/``.

Example 2: Parakeet TDT 0.6b v2 (English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses `model type 30 <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/model_registry.rs>`_ — ``sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8``.
It supports English only.

Step 1: Download the model and Silero VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd tauri-examples/non-streaming-speech-recognition-from-file

   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
   tar xvf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2
   rm sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2

   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

Step 2: Copy assets
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir -p src-tauri/assets
   cp -a sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 src-tauri/assets/
   cp -a silero_vad.onnx src-tauri/assets/

After copying, you can delete the original model directory to save disk space:

.. code-block:: bash

   rm -rf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8

This gives:

.. code-block:: text

   src-tauri/assets/
   ├── silero_vad.onnx
   └── sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/
       ├── encoder.int8.onnx
       ├── decoder.int8.onnx
       ├── joiner.int8.onnx
       └── tokens.txt

Step 3: Set model type
^^^^^^^^^^^^^^^^^^^^^^

Edit `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/lib.rs>`_:

.. code-block:: rust

   const MODEL_TYPE: u32 = 30;
   const MODEL_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

Step 4: Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cargo tauri dev

   # Or build a release binary
   cargo tauri build

Using a Different Model
-----------------------

The app supports 62 models (types 0–61) defined in
`src-tauri/src/model_registry.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/model_registry.rs>`_.
To use a different model:

1. Choose a model type from the registry (0–61).
2. Set ``MODEL_TYPE`` and ``MODEL_NAME`` in `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-file/src-tauri/src/lib.rs>`_.
3. Download the corresponding model and place it in ``src-tauri/assets/``.
4. Always include ``silero_vad.onnx`` in ``src-tauri/assets/``.

The model files required depend on the model family:

- **Paraformer**: ``model.int8.onnx``, ``tokens.txt``
- **Transducer**: ``encoder*.onnx``, ``decoder*.onnx``, ``joiner*.onnx``, ``tokens.txt``
- **Whisper**: ``*-encoder*.onnx``, ``*-decoder*.onnx``, ``tokens.txt``
- **SenseVoice**: ``model.int8.onnx``, ``tokens.txt``
- **Moonshine**: ``encode.onnx``, ``decode.onnx``, ``tokens.txt``
- **NeMo CTC**: ``model.onnx``, ``tokens.txt``

How It Works
------------

1. Click **Select Audio File** to choose an audio/video file via native dialog.
2. The file is decoded by symphonia to mono f32 PCM samples.
3. Audio is resampled to 16 kHz (required by Silero VAD).
4. Audio is fed to the VAD in 512-sample (32 ms) chunks.
5. Each detected speech segment is decoded by the offline recognizer.
6. Results are displayed in a table with start/end timestamps and text.
7. You can export the results as an SRT subtitle file.

License
-------

The code is licensed under
`Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
Please check the license of your selected model separately.
