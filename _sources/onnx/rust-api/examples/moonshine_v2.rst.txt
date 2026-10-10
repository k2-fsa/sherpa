moonshine_v2
============

Run non-streaming ASR with a Moonshine v2 model.

Source file
-----------

`rust-api-examples/examples/moonshine_v2.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/moonshine_v2.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-moonshine-v2.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-moonshine-v2.sh>`_

.. code-block:: bash

  ./run-moonshine-v2.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example moonshine_v2 -- --help
