zipformer
=========

Run non-streaming ASR with a Zipformer transducer model on a WAV file.

Source file
-----------

`rust-api-examples/examples/zipformer.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/zipformer.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-zipformer-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipformer-en.sh>`_

.. code-block:: bash

  ./run-zipformer-en.sh

`run-zipformer-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipformer-zh-en.sh>`_

.. code-block:: bash

  ./run-zipformer-zh-en.sh

`run-zipformer-vi.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-zipformer-vi.sh>`_

.. code-block:: bash

  ./run-zipformer-vi.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example zipformer -- --help

Notes
-----

- The repository provides helper scripts for English, bilingual Chinese-English, and Vietnamese models.
