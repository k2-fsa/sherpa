matcha_tts_en
=============

Run offline text-to-speech with an English Matcha TTS model.

Source file
-----------

`rust-api-examples/examples/matcha_tts_en.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/matcha_tts_en.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-matcha-tts-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-matcha-tts-en.sh>`_

.. code-block:: bash

  ./run-matcha-tts-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example matcha_tts_en
