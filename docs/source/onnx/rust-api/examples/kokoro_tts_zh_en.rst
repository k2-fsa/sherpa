kokoro_tts_zh_en
================

Run offline text-to-speech with a bilingual Chinese-English Kokoro model.

Source file
-----------

`rust-api-examples/examples/kokoro_tts_zh_en.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/kokoro_tts_zh_en.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-kokoro-tts-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-kokoro-tts-zh-en.sh>`_

.. code-block:: bash

  ./run-kokoro-tts-zh-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example kokoro_tts_zh_en
