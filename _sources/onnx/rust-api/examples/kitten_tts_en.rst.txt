kitten_tts_en
=============

Run offline text-to-speech with an English Kitten TTS model.

Source file
-----------

`rust-api-examples/examples/kitten_tts_en.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/kitten_tts_en.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-kitten-tts-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-kitten-tts-en.sh>`_

.. code-block:: bash

  ./run-kitten-tts-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example kitten_tts_en
