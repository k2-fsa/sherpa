online_punctuation
==================

Add punctuation to text with an online punctuation model.

Source file
-----------

`rust-api-examples/examples/online_punctuation.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/online_punctuation.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-online-punctuation.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-online-punctuation.sh>`_

.. code-block:: bash

  ./run-online-punctuation.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example online_punctuation -- --help
