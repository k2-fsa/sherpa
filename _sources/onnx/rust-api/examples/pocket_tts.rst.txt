pocket_tts
==========

Run offline text-to-speech with PocketTTS using zero-shot voice cloning from a reference audio file.

Source file
-----------

`rust-api-examples/examples/pocket_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/pocket_tts.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-pocket-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-pocket-tts.sh>`_

.. code-block:: bash

  ./run-pocket-tts.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example pocket_tts
