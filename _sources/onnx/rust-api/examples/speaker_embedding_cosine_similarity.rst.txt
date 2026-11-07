speaker_embedding_cosine_similarity
===================================

Compute cosine similarity scores between three speaker embeddings.

Source file
-----------

`rust-api-examples/examples/speaker_embedding_cosine_similarity.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/speaker_embedding_cosine_similarity.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-speaker-embedding-cosine-similarity.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-speaker-embedding-cosine-similarity.sh>`_

.. code-block:: bash

  ./run-speaker-embedding-cosine-similarity.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example speaker_embedding_cosine_similarity
