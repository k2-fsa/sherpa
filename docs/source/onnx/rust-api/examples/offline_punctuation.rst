offline_punctuation
===================

Add punctuation to text with an offline punctuation model.

Source file
-----------

`rust-api-examples/examples/offline_punctuation.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/offline_punctuation.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-offline-punctuation.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-offline-punctuation.sh>`_

.. code-block:: bash

  ./run-offline-punctuation.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example offline_punctuation -- --help
