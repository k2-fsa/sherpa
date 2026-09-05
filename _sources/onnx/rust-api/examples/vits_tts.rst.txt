vits_tts
========

Run offline text-to-speech with a standalone VITS/Piper model.

Source file
-----------

`rust-api-examples/examples/vits_tts.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/vits_tts.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-vits-en.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-vits-en.sh>`_

.. code-block:: bash

  ./run-vits-en.sh

`run-vits-de.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-vits-de.sh>`_

.. code-block:: bash

  ./run-vits-de.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example vits_tts -- --help

Notes
-----

- The repository currently provides helper scripts for English and German Piper models.
