supertonic_tts
==============

Run offline text-to-speech with Supertonic TTS for multi-speaker and multi-language synthesis.

Source file
-----------

`rust-api-examples/examples/supertonic_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/supertonic_tts.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-supertonic-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-supertonic-tts.sh>`_

.. code-block:: bash

  ./run-supertonic-tts.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example supertonic_tts
