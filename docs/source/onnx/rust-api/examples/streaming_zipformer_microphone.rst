streaming_zipformer_microphone
==============================

Run real-time streaming ASR from a microphone using a Zipformer model.

Source file
-----------

`rust-api-examples/examples/streaming_zipformer_microphone.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_zipformer_microphone.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-streaming-zipformer-microphone-zh-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-streaming-zipformer-microphone-zh-en.sh>`_

.. code-block:: bash

  ./run-streaming-zipformer-microphone-zh-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --features mic --example streaming_zipformer_microphone -- --help

Notes
-----

- This example requires the ``mic`` Cargo feature.
