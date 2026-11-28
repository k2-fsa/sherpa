zipvoice_tts
============

Run offline text-to-speech with ZipVoice using zero-shot voice cloning.

Source file
-----------

`rust-api-examples/examples/zipvoice_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/zipvoice_tts.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-zipvoice-tts.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipvoice-tts.sh>`_

.. code-block:: bash

  ./run-zipvoice-tts.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example zipvoice_tts
