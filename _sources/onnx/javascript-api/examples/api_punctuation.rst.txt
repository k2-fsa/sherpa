Punctuation API
===============

Punctuation restoration API reference for ``sherpa-onnx-node``.

Source file
-----------

`scripts/node-addon-api/lib/punctuation.js <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/node-addon-api/lib/punctuation.js>`_

API
---

OfflinePunctuation
^^^^^^^^^^^^^^^^^^

Restores punctuation in text (offline, non-streaming).

Constructor
"""""""""""

.. code-block:: javascript

   const punct = new sherpa_onnx.OfflinePunctuation(config);

:param config: Configuration object with:

- ``model`` (object, optional) — Model configuration:

  - ``ctTransformer`` (string) — Path to the CT-Transformer ONNX model.
  - ``numThreads`` (number, optional).
  - ``debug`` (boolean, optional).
  - ``provider`` (string, optional).

Methods
"""""""

``punct.addPunct(text)``
..........................

Add punctuation to the input text.

:param text: Input text without punctuation (``string``).
:returns: Text with punctuation added (``string``).

Properties
""""""""""

- ``punct.config`` — The configuration object.

OnlinePunctuation
^^^^^^^^^^^^^^^^^

Restores punctuation in text (online, streaming).

Constructor
"""""""""""

.. code-block:: javascript

   const punct = new sherpa_onnx.OnlinePunctuation(config);

:param config: Configuration object with:

- ``model`` (object, optional) — Model configuration:

  - ``cnnBilstm`` (string) — Path to the CNN-BiLSTM ONNX model.
  - ``bpeVocab`` (string, optional) — Path to the BPE vocabulary file.
  - ``numThreads`` (number, optional).
  - ``debug`` (boolean, optional).
  - ``provider`` (string, optional).

Methods
"""""""

``punct.addPunct(text)``
..........................

Add punctuation to the input text.

:param text: Input text without punctuation (``string``).
:returns: Text with punctuation added (``string``).

Properties
""""""""""

- ``punct.config`` — The configuration object.

Example
^^^^^^^

.. code-block:: javascript

   const sherpa_onnx = require('sherpa-onnx-node');

   // Offline punctuation
   const punct = new sherpa_onnx.OfflinePunctuation({
     model: { ctTransformer: './punctuation-ct-transformer-zh-en-vocab272727-2024-04-12.onnx' },
   });

   const result = punct.addPunct('今天天气很好我们去公园玩吧');
   console.log(result);
   // Output: '今天天气很好，我们去公园玩吧。'

Notes
-----

- ``OfflinePunctuation`` uses a CT-Transformer model and supports Chinese and
  English text.
- ``OnlinePunctuation`` uses a CNN-BiLSTM model for streaming punctuation.
