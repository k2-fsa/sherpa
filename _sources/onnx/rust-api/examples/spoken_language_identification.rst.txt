spoken_language_identification
==============================

Identify the spoken language in a WAV file using a Whisper-based model.

Source file
-----------

`rust-api-examples/examples/spoken_language_identification.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/spoken_language_identification.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-spoken-language-identification.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-spoken-language-identification.sh>`_

.. code-block:: bash

  ./run-spoken-language-identification.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example spoken_language_identification -- --help
