nemo_parakeet
=============

Run non-streaming ASR with a NeMo Parakeet TDT transducer model.

Source file
-----------

`rust-api-examples/examples/nemo_parakeet.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/nemo_parakeet.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-nemo-parakeet-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-nemo-parakeet-en.sh>`_

.. code-block:: bash

  ./run-nemo-parakeet-en.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example nemo_parakeet -- --help
