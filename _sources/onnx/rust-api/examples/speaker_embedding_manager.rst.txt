speaker_embedding_manager
=========================

Register speakers, search for matches, verify identities, and remove speakers using embeddings.

Source file
-----------

`rust-api-examples/examples/speaker_embedding_manager.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/speaker_embedding_manager.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-speaker-embedding-manager.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-speaker-embedding-manager.sh>`_

.. code-block:: bash

  ./run-speaker-embedding-manager.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example speaker_embedding_manager
