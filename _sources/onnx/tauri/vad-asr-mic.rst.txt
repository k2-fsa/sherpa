.. _tauri-vad-asr-mic:

Non-Streaming Speech Recognition from Microphone
=================================================

This app is a `Tauri v2 <https://v2.tauri.app/>`_ desktop application that
performs **non-streaming speech recognition from live microphone input**.
It uses `Silero VAD <https://github.com/snakers4/silero-vad>`_ to detect
speech segments and then runs an offline ASR model on each segment.

.. note::

   This app captures audio from a **microphone**, not from files.
   It displays wall-clock timestamps, supports recording playback,
   SRT export, and segment WAV export. If you want to transcribe
   audio/video files instead, see :ref:`tauri-vad-asr-file`.

Features
--------

- 62+ ASR models supported (SenseVoice, Paraformer, Whisper, Transducer,
  Moonshine, Parakeet, Canary, Qwen3 ASR, etc.)
- Live microphone capture via `cpal <https://github.com/RustAudioGroup/cpal>`_ —
  cross-platform (ALSA, CoreAudio, WASAPI)
- Microphone device selection — choose from available input devices
- Wall-clock timestamps on each recognized segment
- Recording playback — listen back to the full recording after stopping
- SRT subtitle export
- Segment WAV export — save individual speech segments as WAV files
- Append mode — stop and restart recording without losing previous results
- Cross-platform — macOS (universal), Linux (x64/aarch64), Windows (x64)

Pre-built Apps
--------------

Pre-built apps for 62+ models and 4 platforms are available at:

  `<https://k2-fsa.github.io/sherpa/onnx/tauri/pre-built-app.html#non-streaming-speech-recognition-from-microphone>`_

The filename follows the pattern:

  ``vad-asr-mic-{version}-{lang}-{model}-{platform}.{ext}``

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
- `Node.js <https://nodejs.org/>`_ (for the Tauri CLI)
- `Tauri CLI prerequisites <https://v2.tauri.app/start/prerequisites/>`_
- On Linux, install ALSA development headers: ``sudo apt-get install libasound2-dev``

Install npm dependencies:

.. code-block:: bash

   npm install

Example 1: SenseVoice (multilingual, Chinese/English/Japanese/Korean/Cantonese)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses `model type 15 <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/model_registry.rs>`_ — ``sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17``.
It supports Chinese, English, Japanese, Korean, and Cantonese.

Step 1: Download the model and Silero VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd tauri-examples/non-streaming-speech-recognition-from-microphone

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

Edit `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/lib.rs>`_ and make sure the constants match:

.. code-block:: rust

   const MODEL_TYPE: u32 = 15;
   const MODEL_NAME: &str = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17";

Step 4: Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   npm run dev

To build a release binary:

.. code-block:: bash

   npm run build

The output is in ``src-tauri/target/release/bundle/``.

Example 2: Parakeet TDT 0.6b v2 (English)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses `model type 30 <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/model_registry.rs>`_ — ``sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8``.
It supports English only.

Step 1: Download the model and Silero VAD
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd tauri-examples/non-streaming-speech-recognition-from-microphone

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

Edit `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/lib.rs>`_:

.. code-block:: rust

   const MODEL_TYPE: u32 = 30;
   const MODEL_NAME: &str = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8";

Step 4: Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   npm run dev

   # Or build a release binary
   npm run build

Using a Different Model
-----------------------

The app supports 62+ models (types 0–61) defined in
`src-tauri/src/model_registry.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/model_registry.rs>`_.
To use a different model:

1. Choose a model type from the registry (0–61).
2. Set ``MODEL_TYPE`` and ``MODEL_NAME`` in `src-tauri/src/lib.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/tauri-examples/non-streaming-speech-recognition-from-microphone/src-tauri/src/lib.rs>`_.
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

1. Select a microphone from the device dropdown (or use the default).
2. Click **Start Recording** — the app begins capturing audio from the microphone.
3. Audio is resampled to 16 kHz if needed and fed to the VAD in 512-sample (32 ms) chunks.
4. Each detected speech segment is decoded by the offline recognizer.
5. Results appear in a table with wall-clock timestamps and recognized text.
6. Click **Stop Recording** — remaining audio is flushed through the VAD.
7. The full recording is available for playback. Click any table row to seek to that segment.
8. You can export results as SRT, copy text with timestamps, or save individual segments as WAV.
9. Click **Start Recording** again to append new audio to the existing recording.
10. Click **Clear** to discard all results and recorded audio.

License
-------

The code is licensed under
`Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
Please check the license of your selected model separately.
