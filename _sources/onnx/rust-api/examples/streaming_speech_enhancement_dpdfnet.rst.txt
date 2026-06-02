streaming_speech_enhancement_dpdfnet
====================================

Run streaming speech enhancement with a DPDFNet model.

Source file
-----------

`rust-api-examples/examples/streaming_speech_enhancement_dpdfnet.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/streaming_speech_enhancement_dpdfnet.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-streaming-speech-enhancement-dpdfnet.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-streaming-speech-enhancement-dpdfnet.sh>`_

.. code-block:: bash

  ./run-streaming-speech-enhancement-dpdfnet.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example streaming_speech_enhancement_dpdfnet -- --help
