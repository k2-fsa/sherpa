silero-vad
==========

We support both `silero-vad`_ v4 and v5.

Download models files
----------------------

The following table lists the supported onnx model files of `silero-vad`_
in `sherpa-onnx`_.

.. list-table::

 * -
   - Model size
   - Download URL
   - Comment
 * - | silero_vad.onnx
     | exported by k2-fsa
   - 629 KB
   - `Download <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx>`_
   - | It supports only 16kHz
     | and is exported and maintained by `k2-fsa`_
 * - | silero_vad.int8.onnx
     | exported by k2-fsa
   - 208 KB
   - `Download <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.int8.onnx>`_
   - | It supports only 16kHz
     | and is exported and maintained by `k2-fsa`_
     | It is 8-bit quantized.
 * - silero_vad v4
   - 1.72 MB
   - `Download <https://github.com/snakers4/silero-vad/raw/refs/tags/v4.0/files/silero_vad.onnx>`_
   - It supports both 16kHz and 8kHz samples
 * - silero_vad v5
   - 2.22 MB
   - `Download <https://github.com/snakers4/silero-vad/raw/refs/tags/v5.0/files/silero_vad.onnx>`_
   - It supports both 16kHz and 8kHz samples

If you are curious about how we export the `silero-vad`_ v4 to onnx, you can have a look at

  `<https://github.com/lovemefan/Silero-vad-pytorch/issues/5>`_

We have reverse engineered the PyTorch source code of `silero-vad`_ v4. You can use
it to export `silero-vad`_ to ``onnx``, to executorch, to RKNN, or to torchscript.

Android examples
----------------

.. list-table::

 * -
   - Source code
   - Pre-built APK URL
 * - Pure VAD
   - `Address <https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxVad>`_
   - `Download <https://k2-fsa.github.io/sherpa/onnx/vad/apk.html>`_
 * - VAD + non-streaming ASR
   - `Address <https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxVadAsr>`_
   - `Download <https://k2-fsa.github.io/sherpa/onnx/vad/apk-asr.html>`_
 * - VAD + real-time ASR
   - `Address <https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxSimulateStreamingAsr>`_
   - `Download <https://k2-fsa.github.io/sherpa/onnx/android/apk-simulate-streaming-asr.html>`_


For pure VAD, please see `<https://k2-fsa.github.io/sherpa/onnx/vad/apk.html>`_
and `<https://github.com/k2-fsa/sherpa-onnx/tree/master/android/SherpaOnnxVad>`_

WebAssembly examples
--------------------

.. list-table::

 * -
   - URL
 * - Huggingface space
   - `Address <https://huggingface.co/spaces/k2-fsa/web-assembly-vad-sherpa-onnx>`_
 * - ModelScope space
   - `Address <https://modelscope.cn/studios/csukuangfj/web-assembly-vad-sherpa-onnx>`_

Source code is available at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/wasm/vad>`_

For WebAssembly with VAD + ASR, please see `<https://github.com/k2-fsa/sherpa-onnx/tree/master/wasm/vad-asr>`_

C API examples
--------------

.. list-table::

 * - Filename
   - Comment
 * - `vad-moonshine-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-moonshine-c-api.c>`_
   - | `silero-vad`_ with `moonshine`_ for
     | speech recognition with a very long file
 * - `vad-sense-voice-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-sense-voice-c-api.c>`_
   - | `silero-vad`_ with :ref:`onnx-sense-voice` for
     | speech recognition with a very long file
 * - `vad-whisper-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-whisper-c-api.c>`_
   - | `silero-vad`_ with :ref:`onnx-whisper` for
     | speech recognition with a very long file

For APIs of different programming languages, please see `<https://github.com/k2-fsa/sherpa-onnx>`_
