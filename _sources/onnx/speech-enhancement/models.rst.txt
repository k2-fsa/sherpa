Pre-trained models
==================

Pre-trained models can be found
at `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models>`_

gtcrn_simple
------------

This model is from `<https://github.com/Xiaobin-Rong/gtcrn>`_.
You can find its paper at `<https://ieeexplore.ieee.org/document/10448310>`_.

In the following, we describe how to download and use it with `sherpa-onnx`_.

Download the model
~~~~~~~~~~~~~~~~~~

Please use the following code to download the model:

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx

After downloading, you can check its file size::

  ls -lh  gtcrn_simple.onnx
  -rw-r--r--  1 fangjun  staff   523K Mar 10 18:44 gtcrn_simple.onnx

Then we download a wave file for testing

.. code-block:: bash

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav

.. hint::

   You can find more test wave files at

      `<https://htmlpreview.github.io/?https://github.com/Xiaobin-Rong/gtcrn_demo/blob/main/index.html>`_

The info about the downloaded test wave file is given below::

  soxi ./speech_with_noise.wav

  Input File     : './speech_with_noise.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:02.40 = 38363 samples ~ 179.827 CDDA sectors
  File Size      : 76.8k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM

Now we can run::

    ./build/bin/sherpa-onnx-offline-denoiser \
      --speech-denoiser-gtcrn-model=./gtcrn_simple.onnx \
      --input-wav=./speech_with_noise.wav \
      --output-wav=./enhanced-16k.wav

The log of the above command is::

  /Users/fangjun/open-source/sherpa-onnx/sherpa-onnx/csrc/parse-options.cc:Read:375 ./build/bin/sherpa-onnx-offline-denoiser --speech-denoiser-gtcrn-model=./gtcrn_simple.onnx --input-wav=./speech_with_noise.wav --output-wav=./enhanced-16k.wav

  OfflineSpeechDenoiserConfig(model=OfflineSpeechDenoiserModelConfig(gtcrn=OfflineSpeechDenoiserGtcrnModelConfig(model="./gtcrn_simple.onnx"), num_threads=1, debug=False, provider="cpu"))
  Started
  Done
  Saved to ./enhanced-16k.wav
  num threads: 1
  Elapsed seconds: 0.171 s
  Real time factor (RTF): 0.171 / 2.398 = 0.071

.. code-block:: bash

  ls -lh enhanced-16k.wav
  -rw-r--r--  1 fangjun  staff    75K Mar 22 16:08 enhanced-16k.wav

  soxi ./enhanced-16k.wav

  Input File     : './enhanced-16k.wav'
  Channels       : 1
  Sample Rate    : 16000
  Precision      : 16-bit
  Duration       : 00:00:02.38 = 38144 samples ~ 178.8 CDDA sectors
  File Size      : 76.3k
  Bit Rate       : 256k
  Sample Encoding: 16-bit Signed Integer PCM

For comparison, we give the two wave files below so that you can listen to them.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
    </tr>

    <tr>
      <td>speech_with_noise.wav</td>
      <td>
       <audio title="speech_with_noise.wav" controls="controls">
             <source src="/sherpa/_static/speech-enhancement/gtcrn-simple/speech_with_noise.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>

    <tr>
      <td>enhanced-16k.wav</td>
      <td>
       <audio title="enhanced-16k.wav" controls="controls">
             <source src="/sherpa/_static/speech-enhancement/gtcrn-simple/enhanced-16k.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
    </tr>
  </table>
