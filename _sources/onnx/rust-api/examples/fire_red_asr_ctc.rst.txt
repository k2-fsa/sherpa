fire_red_asr_ctc
================

Run non-streaming ASR with a FireRedASR CTC model for Chinese-English speech recognition.

Source file
-----------

`rust-api-examples/examples/fire_red_asr_ctc.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/fire_red_asr_ctc.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-fire-red-asr-ctc.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-fire-red-asr-ctc.sh>`_

.. code-block:: bash

  ./run-fire-red-asr-ctc.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example fire_red_asr_ctc -- --help
