streaming_zipformer
===================

Run streaming ASR with a Zipformer transducer model on a WAV file.

Source file
-----------

`rust-api-examples/examples/streaming_zipformer.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_zipformer.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-streaming-zipformer-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-streaming-zipformer-en.sh>`_

.. code-block:: bash

  ./run-streaming-zipformer-en.sh

`run-streaming-zipformer-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-streaming-zipformer-zh-en.sh>`_

.. code-block:: bash

  ./run-streaming-zipformer-zh-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example streaming_zipformer -- --help

Notes
-----

- The repository provides helper scripts for both English and bilingual Chinese-English models.
