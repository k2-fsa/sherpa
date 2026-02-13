Dart API for SenseVoice
=======================

This page describes how to use the Dart API to run `SenseVoice`_ models
in `sherpa-onnx`_

Note that we have published the package ``sherpa_onnx`` at `<https://pub.dev/packages/sherpa_onnx>`_.

.. figure:: ./pic/pub-dev.png
   :alt: screenshot of the sherpa-onnx package on pub.dev
   :align: center
   :width: 600

   Screenshot of `sherpa-onnx`_ on ``pub.dev``.

Note that the package supports the following platforms:

  - Android
  - iOS
  - Linux
  - macOS
  - Windows

In the following, we show how to use the pure Dart API to decode files
with `SenseVoice`_ models.

.. code-block:: bash

   cd /tmp

   git clone http://github.com/k2-fsa/sherpa-onnx

   cd sherpa-onnx
   cd dart-api-examples
   cd non-streaming-asr
   dart pub get
   ./run-sense-voice.sh

You should see the following recognition result:

  开饭时间早上9点至下午5点。

Explanations
------------

1. Download the code
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd /tmp

   git clone http://github.com/k2-fsa/sherpa-onnx

In this example, we download `sherpa-onnx`_ and place it inside the directory
``/tmp/``. You can replace ``/tmp/`` with any directory you like.

2. Download the sherpa-onnx package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd sherpa-onnx
   cd dart-api-examples
   cd non-streaming-asr
   dart pub get

The command ``dart pub get`` will download the ``sherpa_onnx`` package automagically
from ``pub.dev``.

You should see something like below after running ``dart pub get``::

  (py38) fangjuns-MacBook-Pro:non-streaming-asr fangjun$ dart pub get
  Resolving dependencies... (1.2s)
  Downloading packages... (33.3s)
    collection 1.18.0 (1.19.0 available)
    lints 3.0.0 (4.0.0 available)
    material_color_utilities 0.8.0 (0.12.0 available)
    meta 1.12.0 (1.15.0 available)
  > sherpa_onnx 1.10.17 (was 1.9.29)
  + sherpa_onnx_android 1.10.17
  + sherpa_onnx_ios 1.10.17
  + sherpa_onnx_linux 1.10.17
  + sherpa_onnx_macos 1.10.17
  + sherpa_onnx_windows 1.10.17
  Changed 6 dependencies!
  4 packages have newer versions incompatible with dependency constraints.
  Try `dart pub outdated` for more information.

3. Run it
^^^^^^^^^

.. code-block:: bash

   ./run-sense-voice.sh

The above script downloads models and run the code automatically.

You can find ``run-sense-voice.sh`` at the following address:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/non-streaming-asr/run-sense-voice.sh>`_

The Dart API example code can be found at:

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/dart-api-examples/non-streaming-asr/bin/sense-voice.dart>`_
