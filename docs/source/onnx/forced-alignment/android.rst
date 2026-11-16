.. _sherpa-onnx-forced-alignment-android:

Android
=======

In this section, we describe the Android app for forced alignment. The source
code is in
`android/SherpaOnnxForcedAlignment <https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxForcedAlignment>`_.

.. hint::

   It runs locally on your phone, without internet connection. No audio is
   uploaded anywhere.

The app takes an audio/video file (or a microphone recording) and its
transcript and shows a table of the words with their start/end times. You can

  - click a row to play just that word (sample-accurately cut from the audio)
  - export the result as a Praat ``TextGrid`` or as txt files
  - share the result with other apps (WeChat, email, ...)
  - try the whole flow with the bundled example audio

Pre-built APKs
--------------

One APK per model and per ABI. Download the APK of your model and ABI from
the `apk-fa <https://github.com/k2-fsa/sherpa-onnx/releases/tag/apk-fa>`_
release, e.g. for 64-bit ARM phones:

.. list-table::

 * - Model
   - Language
   - APK
 * - ``zh-aishell3-pinyin``
   - Chinese
   - `sherpa-onnx-fa-arm64-v8a-zh-aishell3.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-zh-aishell3.apk>`_
 * - ``zh-biaobei-pinyin``
   - Chinese
   - `sherpa-onnx-fa-arm64-v8a-zh-biaobei.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-zh-biaobei.apk>`_
 * - ``zh-large-pinyin``
   - Chinese
   - `sherpa-onnx-fa-arm64-v8a-zh-large.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-zh-large.apk>`_
 * - ``english-mfa-v2.2.1``
   - English
   - `sherpa-onnx-fa-arm64-v8a-en-v2.2.1.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-en-v2.2.1.apk>`_
 * - ``english-mfa-v3.0.0``
   - English
   - `sherpa-onnx-fa-arm64-v8a-en-v3.0.0.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-en-v3.0.0.apk>`_
 * - ``english-mfa-v3.1.0``
   - English
   - `sherpa-onnx-fa-arm64-v8a-en-v3.1.0.apk <https://github.com/k2-fsa/sherpa-onnx/releases/download/apk-fa/sherpa-onnx-fa-arm64-v8a-en-v3.1.0.apk>`_

The same models exist for the ABIs ``armeabi-v7a``, ``x86_64`` and ``x86`` —
replace ``arm64-v8a`` in the file name with your ABI (``arm64-v8a`` is right
for most modern phones). See
`scripts/mfa/android.md <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/mfa/android.md>`_
for the full table of APKs and how to choose one.

Build from source
-----------------

Please see
`android/SherpaOnnxForcedAlignment/README.md <https://github.com/k2-fsa/sherpa-onnx/blob/master/android/SherpaOnnxForcedAlignment/README.md>`_
for how to build the app yourself.
