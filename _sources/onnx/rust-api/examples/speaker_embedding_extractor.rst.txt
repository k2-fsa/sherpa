speaker_embedding_extractor
===========================

Compute a speaker embedding vector from a WAV file.

Source file
-----------

`rust-api-examples/examples/speaker_embedding_extractor.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/speaker_embedding_extractor.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-speaker-embedding-extractor.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-speaker-embedding-extractor.sh>`_

.. code-block:: bash

  ./run-speaker-embedding-extractor.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example speaker_embedding_extractor
