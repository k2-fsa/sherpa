audio_tagging_ced
=================

Run audio tagging with a CED model.

Source file
-----------

`rust-api-examples/examples/audio_tagging_ced.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/audio_tagging_ced.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-audio-tagging-ced.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-audio-tagging-ced.sh>`_

.. code-block:: bash

  ./run-audio-tagging-ced.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example audio_tagging_ced
