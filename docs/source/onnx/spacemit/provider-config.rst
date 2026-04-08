.. _spacemit-provider-config:

Provider configuration for SpacemiT
===================================

For the complete and authoritative provider option list, please refer to:

`SpacemiT-ONNXRuntime ProviderOption documentation <https://github.com/spacemit-com/docs-ai/blob/main/zh/compute_stack/ai_compute_stack/onnxruntime.md>`_

This page only keeps the usage pattern in `sherpa-onnx`_ and a few commonly
used options.

Basic usage
-----------

The simplest form is:

.. code-block:: bash

   --provider=spacemit

`sherpa-onnx`_ also supports loading a provider config file by appending a
path after ``:``:

.. code-block:: bash

   --provider=spacemit:path/to/provider.config

The provider string is split into two parts:

- The provider name, for example ``spacemit``
- The config file path after ``:``

Config file format
------------------

The config file is a plain text file.

- Empty lines are ignored
- Lines starting with ``#`` are ignored
- Each entry uses ``key=value`` format or ``key = value``

For example:

.. code-block:: text

   # Example SpacemiT provider config
   SPACEMIT_EP_INTRA_THREAD_NUM=4

Common options
--------------

Please check the external SpacemiT-ONNXRuntime documentation for the full
description. The options below are usually the most relevant when using
`sherpa-onnx`_.

``SPACEMIT_EP_INTRA_THREAD_NUM``
   Controls the EP intra-thread count. This is the first option to tune when
   you want to adjust CPU utilization.

``SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD``
   Lets multiple sessions share the same EP intra-thread pool in one process.
   This is useful only when you clearly control session scheduling.

``SPACEMIT_EP_DUMP_SUBGRAPHS``
   Dumps EP subgraphs for debugging graph partition behavior.

``SPACEMIT_EP_DEBUG_PROFILE``
   Writes EP profiling data that can be inspected with timeline tools.

``SPACEMIT_EP_DUMP_TENSORS``
   Dumps intermediate tensors for debugging correctness issues.

Threading behavior
------------------

When the SpacemiT execution provider is enabled, `sherpa-onnx`_ adjusts ORT
threading internally:

- Intra-op threads are set to ``1``
- Inter-op threads are set to ``1``
- If ``SPACEMIT_EP_INTRA_THREAD_NUM`` is not present in the provider config,
  `sherpa-onnx`_ uses ``--num-threads`` to populate it automatically

This means the following command is valid even without an explicit config file:

.. code-block:: bash

   sherpa-onnx-offline --provider=spacemit --num-threads=4 ...

Common recommendations
----------------------

- Start with ``--provider=spacemit`` and ``--num-threads=N`` before creating a
   custom config file.
- Add a config file only when you need explicit EP tuning or debugging.
- Use ``SPACEMIT_EP_DUMP_SUBGRAPHS`` and ``SPACEMIT_EP_DEBUG_PROFILE`` when you
   investigate partition or performance issues.
- Use ``SPACEMIT_EP_DISABLE_OP_TYPE_FILTER`` for temporary compatibility
   workarounds when a specific operator type should fall back to CPU.

Starter template
----------------

You can create a minimal provider config file like this:

.. code-block:: text

   SPACEMIT_EP_INTRA_THREAD_NUM=4
   SPACEMIT_EP_USE_GLOBAL_INTRA_THREAD=1
