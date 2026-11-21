.. _onnx-spacemit-cpu:

SpacemiT CPUs
==============

This section explains how to use `sherpa-onnx`_ with the SpacemiT execution
provider on SpacemiT RISC-V CPUs.

At the moment, this section is a starter skeleton. We focus on the following
topics first:

- Building `sherpa-onnx`_ with SpacemiT provider enabled
- Running ASR and TTS examples with ``--provider=spacemit``
- Using a provider config file through ``--provider=spacemit:path/to/config``

The current build and example commands in this section are based on the
existing `sherpa-onnx`_ build scripts and CI workflow.

The following boards are known to work:

  - SpacemiT K1(Keystone X60+A60)
  - SpacemiT K3(Keystone X100+A100)
  - Another board with a SpacemiT RISC-V CPU (to be added later)

.. toctree::
   :maxdepth: 8

   ./build.rst
   ./provider-config.rst
   ./examples.rst
