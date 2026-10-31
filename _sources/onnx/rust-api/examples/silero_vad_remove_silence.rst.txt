silero_vad_remove_silence
=========================

Use Silero VAD to remove non-speech segments from a WAV file.

Source file
-----------

`rust-api-examples/examples/silero_vad_remove_silence.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/silero_vad_remove_silence.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-silero-vad-remove-silence.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-silero-vad-remove-silence.sh>`_

.. code-block:: bash

  ./run-silero-vad-remove-silence.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example silero_vad_remove_silence -- --help
