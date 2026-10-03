ten-vad
==========

.. caution::

   Please see its license at

    `<https://github.com/TEN-framework/ten-vad/blob/main/LICENSE>`_

  before you use it ``commercially``.

  如果你需要把它用于商业目的，请先阅读它的 `协议 <https://github.com/TEN-framework/ten-vad/blob/main/LICENSE>`_ 。

Our support of `ten-vad`_ uses `<https://github.com/TEN-framework/ten-vad/pull/36>`_
as a reference, which use 0 for the pitch feature. It may degrade the performance,
but it greatly simplifies the implementation.

Download models files
----------------------

We have added some meta data to the original ``ten-vad.onnx``, so please
use the model files from the following table:

.. list-table::

 * -
   - Model size
   - Download URL
 * - ``ten-vad.onnx``
   - 324 KB
   - `Download <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx>`_
 * - ``ten-vad.int8.onnx``
   - 126 KB
   - `Download <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.int8.onnx>`_

If you use the model from `<https://github.com/TEN-framework/ten-vad/blob/main/src/onnx_model/ten-vad.onnx>`_
in `sherpa-onnx`_, you will get runtime errors.

Note that `ten-vad`_ supports only 16k Hz samples.

Android examples
----------------

.. list-table::

 * -
   - Source
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
   - `Address <https://huggingface.co/spaces/k2-fsa/web-assembly-ten-vad-sherpa-onnx>`_
 * - ModelScope space
   - `Address <https://modelscope.cn/studios/csukuangfj/web-assembly-ten-vad-sherpa-onnx>`_

Source code is available at `<https://github.com/k2-fsa/sherpa-onnx/tree/master/wasm/vad>`_

For WebAssembly with VAD + ASR, please see `<https://github.com/k2-fsa/sherpa-onnx/tree/master/wasm/vad-asr>`_

C API examples
--------------

.. list-table::

 * - Filename
   - Comment
 * - `vad-moonshine-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-moonshine-c-api.c>`_
   - | `ten-vad`_ with `moonshine`_ for
     | speech recognition with a very long file
 * - `vad-sense-voice-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-sense-voice-c-api.c>`_
   - | `ten-vad`_ with :ref:`onnx-sense-voice` for
     | speech recognition with a very long file
 * - `vad-whisper-c-api.c <https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/vad-whisper-c-api.c>`_
   - | `ten-vad`_ with :ref:`onnx-whisper` for
     | speech recognition with a very long file

For APIs of different programming languages, please see `<https://github.com/k2-fsa/sherpa-onnx>`_
