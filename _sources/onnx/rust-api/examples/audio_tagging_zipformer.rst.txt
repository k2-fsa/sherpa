audio_tagging_zipformer
=======================

Run audio tagging with a Zipformer-based tagging model.

Source file
-----------

`rust-api-examples/examples/audio_tagging_zipformer.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/audio_tagging_zipformer.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-audio-tagging-zipformer.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-audio-tagging-zipformer.sh>`_

.. code-block:: bash

  ./run-audio-tagging-zipformer.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example audio_tagging_zipformer
