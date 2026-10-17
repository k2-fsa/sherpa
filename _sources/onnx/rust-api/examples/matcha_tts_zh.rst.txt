matcha_tts_zh
=============

Run offline text-to-speech with a Chinese Matcha TTS model.

Source file
-----------

`rust-api-examples/examples/matcha_tts_zh.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/matcha_tts_zh.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-matcha-tts-zh.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-matcha-tts-zh.sh>`_

.. code-block:: bash

  ./run-matcha-tts-zh.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example matcha_tts_zh
