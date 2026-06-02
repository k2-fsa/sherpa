sense_voice
===========

Run non-streaming ASR with a SenseVoice model supporting Chinese, English, Japanese, Korean, and Cantonese.

Source file
-----------

`rust-api-examples/examples/sense_voice.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/sense_voice.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-sense-voice.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-sense-voice.sh>`_

.. code-block:: bash

  ./run-sense-voice.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example sense_voice -- --help
