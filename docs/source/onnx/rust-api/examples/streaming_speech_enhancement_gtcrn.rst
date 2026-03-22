streaming_speech_enhancement_gtcrn
==================================

Run streaming speech enhancement with a GTCRN model.

Source file
-----------

`rust-api-examples/examples/streaming_speech_enhancement_gtcrn.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_speech_enhancement_gtcrn.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-streaming-speech-enhancement-gtcrn.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-streaming-speech-enhancement-gtcrn.sh>`_

.. code-block:: bash

  ./run-streaming-speech-enhancement-gtcrn.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example streaming_speech_enhancement_gtcrn -- --help
