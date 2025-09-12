MMS
===

This section describes how to convert models
from `<https://huggingface.co/facebook/mms-tts/tree/main>`_
to `sherpa-onnx`_.

Note that `facebook/mms-tts <https://huggingface.co/facebook/mms-tts/tree/main>`_
supports more than 1000 languages. You can try models from
`facebook/mms-tts <https://huggingface.co/facebook/mms-tts/tree/main>`_ at
the huggingface space `<https://huggingface.co/spaces/mms-meta/MMS>`_.

You can try the converted models by visiting `<https://huggingface.co/spaces/k2-fsa/text-to-speech>`_.
To download the converted models, please visit `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_.
If a filename contains ``vits-mms``, it means the model is from
`facebook/mms-tts <https://huggingface.co/facebook/mms-tts/tree/main>`_.

Install dependencies
--------------------

.. code-block:: bash

  pip install -qq onnx scipy Cython
  pip install -qq torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

Download the model file
-----------------------

Suppose that we want to convert the English model, we need to
use the following commands to download the model:

.. code-block:: bash

    name=eng
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/G_100000.pth
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/config.json
    wget -q https://huggingface.co/facebook/mms-tts/resolve/main/models/$name/vocab.txt

Download MMS source code
------------------------

.. code-block:: bash

  git clone https://huggingface.co/spaces/mms-meta/MMS
  export PYTHONPATH=$PWD/MMS:$PYTHONPATH
  export PYTHONPATH=$PWD/MMS/vits:$PYTHONPATH

  pushd MMS/vits/monotonic_align

  python3 setup.py build

  ls -lh build/
  ls -lh build/lib*/
  ls -lh build/lib*/*/

  cp build/lib*/vits/monotonic_align/core*.so .

  sed -i.bak s/.monotonic_align.core/.core/g ./__init__.py
  popd

Convert the model
-----------------

Please save the following code into a file with name ``./vits-mms.py``:

.. literalinclude:: ./code/vits-mms.py

The you can run it with:

.. code-block:: bash

   export PYTHONPATH=$PWD/MMS:$PYTHONPATH
   export PYTHONPATH=$PWD/MMS/vits:$PYTHONPATH
   export lang=eng
   python3 ./vits-mms.py

It will generate the following two files:

  - ``model.onnx``
  - ``tokens.txt``

Use the converted model
-----------------------

We can use the converted model with the following command after installing
`sherpa-onnx`_.

.. code-block:: bash

  ./build/bin/sherpa-onnx-offline-tts \
    --vits-model=./model.onnx \
    --vits-tokens=./tokens.txt \
    --debug=1 \
    --output-filename=./mms-eng.wav \
    "How are you doing today? This is a text-to-speech application using models from facebook with next generation Kaldi"

The above command should generate a wave file ``mms-eng.wav``.

.. raw:: html

  <table>
    <tr>
      <th>Wave filename</th>
      <th>Content</th>
      <th>Text</th>
    </tr>
    <tr>
      <td>mms-eng.wav</td>
      <td>
       <audio title="Generated ./mms-eng.wav" controls="controls">
             <source src="/sherpa/_static/mms/mms-eng.wav" type="audio/wav">
             Your browser does not support the <code>audio</code> element.
       </audio>
      </td>
      <td>
        How are you doing today? This is a text-to-speech application using models from facebook with next generation Kaldi
      </td>
    </tr>
  </table>


Congratulations! You have successfully converted a model from `MMS`_ and run it with `sherpa-onnx`_.

We are using ``eng`` in this section as an example, you can replace it with other languages, such as
``deu`` for German, ``fra`` for French, etc.
