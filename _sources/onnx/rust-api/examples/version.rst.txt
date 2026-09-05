version
=======

Print the ``sherpa-onnx`` version, Git SHA1, and Git date.

Source file
-----------

`rust-api-examples/examples/version.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/version.rs>`_

How to run
----------

The recommended way is to use the helper script(s) provided in
``rust-api-examples`` because they download or point to the required models and test files automatically when needed.

Helper script(s)
^^^^^^^^^^^^^^^^

`run-version.sh <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/run-version.sh>`_

.. code-block:: bash

  ./run-version.sh

Run it directly with Cargo
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  cargo run --example version

Notes
-----

- This is the simplest way to verify that your Rust installation is working correctly.
