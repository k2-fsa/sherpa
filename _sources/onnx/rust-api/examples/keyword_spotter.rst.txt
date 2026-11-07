keyword_spotter
===============

Detect keywords from audio with a Zipformer keyword spotting model.

Source file
-----------

`rust-api-examples/examples/keyword_spotter.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/keyword_spotter.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-keyword-spotter.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-keyword-spotter.sh>`_

.. code-block:: bash

  ./run-keyword-spotter.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example keyword_spotter -- --help
