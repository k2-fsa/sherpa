Pre-built Tauri Apps
====================

Links for pre-built Apps can be found in the following tables.

.. hint::

   It runs locally, without internet connection.

Non-Streaming Speech Recognition from File
------------------------------------------

.. hint::

   You can use it to generate subtitles for Audio and Video files.

.. hint::

   The code is available at

      `<https://github.com/k2-fsa/sherpa-onnx/tree/master/tauri-examples/non-streaming-speech-recognition-from-file>`_

.. list-table::

 * - ****
   - Chinese users
   - URL
 * - All platforms
   - `地址 <https://k2-fsa.github.io/sherpa/onnx/tauri/app/vad-asr-file-cn.html>`_
   - `Here <https://k2-fsa.github.io/sherpa/onnx/tauri/app/vad-asr-file.html>`_


.. hint::

   The pre-built APPs support Linux (x64, aarch64), macOS (x64, arm64), and Windows (x64).

   You can use the following files to test the APPs:

    - Chinese audio: `lei-jun-test.wav <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav>`_
    - Chinese video: `lei-jun-test.mov <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.mov>`_
    - English audio: `Obama.wav <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav>`_
    - English video: `Obama.mov <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.mov>`_

Screenshots
~~~~~~~~~~~

.. list-table::
   :align: center

   * - .. image:: ./pic/vad-asr-file-1.png
          :width: 200px
     - .. image:: ./pic/vad-asr-file-2.png
          :width: 200px
     - .. image:: ./pic/vad-asr-file-3.png
          :width: 200px

Video demo
~~~~~~~~~~

.. raw:: html

  <iframe src="//player.bilibili.com/player.html?bvid=BV1cXoKBhEdz&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="600" height="600"> </iframe>


Notes for macOS users
~~~~~~~~~~~~~~~~~~~~~

After double click the ``app``, if you get the following dialog:

  .. figure:: ./pic/1-done.jpg
     :alt: Click Done
     :width: 250

     Step 1: Click ``Done``.

Please click ``Done``. And then start your ``Settings``, click ``Privacy & Security``, and then click
``Open Anyway``, as shown below:

  .. figure:: ./pic/2-open.jpg
     :alt: Click Done
     :width: 250

     Step 2: Click ``Open Anyway``.

It will pop a new window, click ``Open Anyway``, as shown below:

  .. figure:: ./pic/3-open.jpg
     :alt: Click Done
     :width: 250

     Step 3: Click ``Open Anyway``.

Click ``Open Anyway`` and enter your password. You should see the following screenshot:

  .. figure:: ./pic/4-started.jpg
     :alt: Click Done
     :width: 350

     Step 4: Started
