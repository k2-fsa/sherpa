Piper
=====

In this section, we describe how to convert `piper`_ pre-trained models
from `<https://huggingface.co/rhasspy/piper-voices>`_.

.. hint::

   You can find ``all`` of the converted models from `piper`_ in the following address:

    `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_

  If you want to convert your own pre-trained `piper`_ models or if you want to
  learn how the conversion works, please read on.

  Otherwise, you only need to download the converted models from the above link.

Note that there are pre-trained models for over 30 languages from `piper`_. All models
share the same converting method, so we use an American English model in this
section as an example.

Install dependencies
--------------------

.. code-block:: bash

   pip install onnx onnxruntime

.. hint::

   We suggest that you always use the latest version of onnxruntime.

Find the pre-trained model from piper
-------------------------------------

All American English models from `piper`_ can be found at
`<https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US>`_.

We use `<https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/amy/low>`_ as
an example in this section.

Download the pre-trained model
------------------------------

We need to download two files for each model:

.. code-block:: bash

   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/low/en_US-amy-low.onnx.json

Add meta data to the onnx model
-------------------------------

Please use the following code to add meta data to the downloaded onnx model.

.. literalinclude:: ./code/piper.py
   :language: python

After running the above script, your ``en_US-amy-low.onnx`` is updated with
meta data and it also generates a new file ``tokens.txt``.

From now on, you don't need the config json file ``en_US-amy-low.onnx.json`` any longer.

Download espeak-ng-data
-----------------------

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
   tar xf espeak-ng-data.tar.bz2

Note that ``espeak-ng-data.tar.bz2`` is shared by all models from `piper`_, no matter
which language your are using for your model.

Test your converted model
-------------------------

To have a quick test of your converted model, you can use

.. code-block:: bash

   pip install sherpa-onnx

to install `sherpa-onnx`_ and then use the following commands to test your model:

.. code-block:: bash

   # The command "pip install sherpa-onnx" will install several binaries,
   # including the following one

   which sherpa-onnx-offline-tts

   sherpa-onnx-offline-tts \
     --vits-model=./en_US-amy-low.onnx \
     --vits-tokens=./tokens.txt \
     --vits-data-dir=./espeak-ng-data \
     --output-filename=./test.wav \
     "How are you doing? This is a text-to-speech application using next generation Kaldi."

The above command should generate a wave file ``test.wav``.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>test.wav</td>
      <td>
       <audio title="Generated ./test.wav" controls="controls">
             <source src="/sherpa/_static/piper/test.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        How are you doing? This is a text-to-speech application using next generation Kaldi.
      </td>
    </tr>
  </table>


Congratulations! You have successfully converted a model from `piper`_ and run it with `sherpa-onnx`_.


