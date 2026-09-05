Offline Punctuation
====================

Add punctuation to unpunctuated text using a CT-Transformer model. This is
useful for post-processing ASR output or restoring punctuation in raw text.

Source file
-----------

`nodejs-addon-examples/test_offline_punctuation.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/nodejs-addon-examples/test_offline_punctuation.js>`_

Code
----

.. literalinclude:: ../code/offline_punctuation.js
   :language: javascript
   :linenos:

How to run
----------

1. Install the package::

     npm install sherpa-onnx-node

2. Download the model::

     curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
     tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
     rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

3. Set the library path and run:

   .. code-block:: bash

      # macOS
      export DYLD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$DYLD_LIBRARY_PATH

      # Linux
      export LD_LIBRARY_PATH=$(npm root)/sherpa-onnx-node/lib:$LD_LIBRARY_PATH

      node offline_punctuation.js

Expected output
^^^^^^^^^^^^^^^

.. code-block:: text

   ---
   Input:  这是一个测试你好吗How are you我很好thank you are you ok谢谢你
   Output: 这是一个测试，你好吗？How are you? 我很好，thank you, are you ok? 谢谢你。
   ---
   Input:  我们都是木头人不会说话不会动
   Output: 我们都是木头人，不会说话，不会动。
   ---
   Input:  The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry
   Output: The African blogosphere is rapidly expanding, bringing more voices online in the form of commentaries, opinions, analyses, rants, and poetry.
   ---

Notes
-----

- The model supports both Chinese and English text.
- ``addPunct()`` takes a single string and returns the punctuated version.
- The model handles mixed Chinese-English text correctly.
