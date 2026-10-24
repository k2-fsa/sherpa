.. _install_sherpa_onnx_rust:

Install the Rust crate
======================

For most users, the recommended setup is the default one:

- depend on ``sherpa-onnx`` normally
- use the crate's default ``static`` feature
- let the build script download matching native libraries automatically

You usually do not need to configure Rust linking details manually.

Add it to your project
----------------------

Use the default dependency in your ``Cargo.toml``:

.. parsed-literal::

  [dependencies]
  |sherpa_onnx_crate_dependency|

The crate's default feature set enables ``static`` linking.

.. hint::

   During the first build, the matching native ``sherpa-onnx`` libraries may be
   downloaded automatically for your platform. This is the expected behavior for
   the default Rust setup.

Check that the installation works
---------------------------------

We provide a small Rust example that prints the ``sherpa-onnx`` version.

First, clone the repository and enter the Rust examples directory:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx/rust-api-examples

Then run:

.. code-block:: bash

  cargo run --example version

It should print something like:

.. parsed-literal::

  |sherpa_onnx_version_output|
  Git SHA1: ...
  Git date: ...

The ``version`` example is implemented in
`rust-api-examples/examples/version.rs <https://github.com/k2-fsa/sherpa-onnx/blob/master/rust-api-examples/examples/version.rs>`_
and prints:

.. code-block:: rust

  println!("Version : {}", sherpa_onnx::version());
  println!("Git SHA1: {}", sherpa_onnx::git_sha1());
  println!("Git date: {}", sherpa_onnx::git_date());

You can also use the helper script:

.. code-block:: bash

  ./run-version.sh

Once this works, your Rust installation is ready to run the other examples in
``rust-api-examples``.

Where to go next
----------------

Please see
`rust-api-examples <https://github.com/k2-fsa/sherpa-onnx/tree/master/rust-api-examples>`_
for more examples.

If you want to use shared libraries, provide your own native library directory,
or enable microphone examples, please see :ref:`install_sherpa_onnx_rust_advanced`.
