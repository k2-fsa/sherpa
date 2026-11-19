.. _sherpa-onnx-forced-alignment-web:

Web demo
========

In this section, we describe the web demo for forced alignment. It is a
pure-browser app running on `WebAssembly`_: pick an audio/video file (or
record from the microphone), enter its transcript, and get the start/end
time of every word.

.. hint::

   It runs locally in your browser, without a backend. No audio is uploaded
   anywhere.

The source code is in
`react-examples/forced-alignment <https://github.com/k2-fsa/sherpa-onnx/tree/master/react-examples/forced-alignment>`_
(the WebAssembly module is in
`wasm/forced-alignment <https://github.com/k2-fsa/sherpa-onnx/tree/master/wasm/forced-alignment>`_).

Try it in your browser
----------------------

One demo per model; click the URL of the model you want to use. The first
visit downloads the model (3–37 MB) with a progress bar; later visits use
the browser cache.

.. list-table::

 * - Model
   - Language
   - Entry point
 * - ``zh-aishell3-pinyin``
   - Chinese
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-aishell3-pinyin/>`_
 * - ``zh-biaobei-pinyin``
   - Chinese
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-biaobei-pinyin/>`_
 * - ``zh-large-pinyin``
   - Chinese
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/zh-large-pinyin/>`_
 * - ``english-mfa-v2.2.1``
   - English
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v2.2.1/>`_
 * - ``english-mfa-v3.0.0``
   - English
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.0.0/>`_
 * - ``english-mfa-v3.1.0``
   - English
   - `<https://k2-fsa.github.io/sherpa/onnx/forced-alignment/demo/english-mfa-v3.1.0/>`_

The same demos can be downloaded from the
`react-fa <https://github.com/k2-fsa/sherpa-onnx/releases/tag/react-fa>`_
release as ``sherpa-onnx-react-fa-<model>.zip`` — unzip a file and serve the
directory with any static HTTP server to host it yourself. See
`scripts/mfa/web.md <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/mfa/web.md>`_
for a description of each file.

Build from source
-----------------

Please see
`react-examples/forced-alignment/README.md <https://github.com/k2-fsa/sherpa-onnx/blob/master/react-examples/forced-alignment/README.md>`_
for how to build the wasm module and the web app yourself.
