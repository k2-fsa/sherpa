How to change the package name of our kotlin and/or java API
============================================================

By default, we use:

.. code-block:: kotlin

   package com.k2fsa.sherpa.onnx


If you change our package name without changing the JNI C++ binding code, you would
get errors like:

  - `<https://github.com/k2-fsa/sherpa-onnx/issues/1994>`_ changes our package
    name from ``com.k2fsa.sherpa.onnx`` to ``stt`` and gets the following error:

      .. code-block::

          No implementation found for
          long stt.OfflineRecognizer.newFromAsset(android.content.res.AssetManager, stt.OfflineRecognizerConfig)
          (tried Java_stt_OfflineRecognizer_newFromAsset and
          Java_stt_OfflineRecognizer_newFromAsset__Landroid_content_res_AssetManager_2Lstt_OfflineRecognizerConfig_2) -
          is the library loaded, e.g.  System.loadLibrary?

We suggest that you don't change our package name when using our code. You can use ``import``
to use our Kotlin or Java API.

If you are familiar with JNI and really want to change our package name, please have a look
at:

  `<https://github.com/mjnong/sherpa-onnx-qnn/pull/1>`_

It shows how to change the package name from ``com.k2fsa.sherpa.onnx`` to ``com.edgeai.chatappv2``.

.. warning::

   You need to change a lot of files.
