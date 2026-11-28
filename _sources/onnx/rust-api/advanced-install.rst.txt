.. _install_sherpa_onnx_rust_advanced:

Advanced installation
=====================

This page is for users who want to control how the Rust crate finds and links
its native ``sherpa-onnx`` libraries.

Most users do **not** need anything here. The default behavior is:

1. Build normally.
2. Let the build script download the matching native libraries automatically.
3. Run your Rust program or one of the examples.

Use shared libraries
--------------------

The default Rust configuration uses ``static`` linking. If you want shared
libraries instead, disable default features and enable ``shared``:

.. parsed-literal::

  [dependencies]
  |sherpa_onnx_crate_shared_dependency|

You can verify this setup with:

.. code-block:: bash

  cd sherpa-onnx/rust-api-examples
  cargo run --no-default-features --features shared --example version

Use your own sherpa-onnx libraries
----------------------------------

If you already have ``sherpa-onnx`` libraries on disk, set
``SHERPA_ONNX_LIB_DIR`` to the ``lib`` directory before building:

.. code-block:: bash

  export SHERPA_ONNX_LIB_DIR=/path/to/sherpa-onnx/lib

Examples:

.. parsed-literal::

  /path/to/sherpa-onnx/build/install/lib
  |sherpa_onnx_linux_static_lib_dir|

If ``SHERPA_ONNX_LIB_DIR`` is set, the build script uses that directory and
does not auto-download another archive.

Automatic download behavior
---------------------------

If ``SHERPA_ONNX_LIB_DIR`` is not set, ``sherpa-onnx-sys`` downloads a matching
prebuilt ``-lib`` archive from GitHub releases and uses its ``lib`` directory
automatically.

Default mode uses the crate's default feature set, which means ``static``
linking. If you enable the ``shared`` feature, shared archives are downloaded
instead.

Enable microphone examples
--------------------------

In ``rust-api-examples``, microphone support is controlled by the ``mic``
feature:

.. code-block:: bash

  cd sherpa-onnx/rust-api-examples
  cargo run --features mic --example streaming_zipformer_microphone -- --help

If you want both microphone support and shared libraries:

.. code-block:: bash

  cargo run --no-default-features --features "shared,mic" \
    --example streaming_zipformer_microphone -- --help

Runtime notes for shared builds
-------------------------------

When shared libraries are used:

- Linux and macOS: the build script adds runtime rpath entries automatically
- Linux and macOS: the required shared libraries are copied next to Cargo-generated binaries and examples
- Windows: the required DLLs are copied next to the generated binaries automatically

When ``SHERPA_ONNX_LIB_DIR`` is set, the same behavior applies, but the files
come from your directory instead of an auto-downloaded archive.
