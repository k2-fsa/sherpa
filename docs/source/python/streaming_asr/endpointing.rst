Endpointing
===========

We borrow the idea of online endpointing from Kaldi.
Please see
`<https://kaldi-asr.org/doc/structkaldi_1_1OnlineEndpointRule.html>`_.

By endpointing in this context we mean "deciding when to stop decoding", and not
generic speech/silence segmentation. The use-case that we have in mind is some
kind of dialog system where, as more speech data comes in, we decode more
and more, and we have to decide when to stop decoding.

Whenever an endpoint is detected, the server sends the following message
to the client:

.. code-block:: python

  message = {
      "segment": stream.segment,
      "text": recogntion results of this segment,
      "final": True,
  }

where ``stream.segment`` is an integer and it is incremented whenever an endpoint
is detected.

The endpointing rule is a disjunction of conjunctions. The way we have it configured,
it's an ``OR`` of ``3`` rules, and each rule has the following form:

.. code-block::

   (<contains-nonsilence> || !rule.must_contain_nonsilence) && \
   <length-of-trailing-silence> >= rule.min_trailing_silence && \
   <utterance-length> >= rule.min_utterance_length)

where:
  - ``<contains-nonsilence>`` is true if the best traceback contains
    any non-blank tokens
  - ``<length-of-trailing-silence>`` is the length in seconds of blank
    token at the end of the best traceback (we stop counting when we hit
    non-blank tokens)
  - ``<utterance-length>`` is the number of seconds of the utterance that we have decoded so far

All of these pieces of information are obtained from the best-path traceback from the decoder.
We do this every time we're finished processing a chunk of data.

For details of the default rules, see the following file:

`<https://github.com/k2-fsa/sherpa/blob/master/sherpa/python/sherpa/online_endpoint.py>`_

In summary, we have the following 3 default rules. If any of them is activated,
we say that an endpoint is detected.

  - ``Rule 1``: It times out after 5 seconds of silence, even if we decoded nothing

    You can use the following commandline arguments to change it:

    .. code-block::

        --endpoint.rule1.must-contain-nonsilence=false \
        --endpoint.rule1.min-trailing-silence=5.0 \
        --endpoint.rule1.min-utterance-length=0.0

  - ``Rule 2``: It times out after 2.0 seconds of silence after decoding something

    You can use the following commandline arguments to change it:

    .. code-block::

        --endpoint.rule2.must-contain-nonsilence=true \
        --endpoint.rule2.min-trailing-silence=2.0 \
        --endpoint.rule2.min-utterance-length=0.0

  - ``Rule 3``: It times out after the utterance is 20 seconds long, regardless of anything else

    You can use the following commandline arguments to change it:

    .. code-block::

        --endpoint.rule3.must-contain-nonsilence=false
        --endpoint.rule3.min-trailing-silence=0.0
        --endpoint.rule3.min-utterance-length=20.0

Endpointing Demo (English)
--------------------------

The following video shows an endpointing demo using a pretrained model
on the `LibriSpeech`_ dataset. You can find the usage in
:ref:`lstm_server_english`.

..  youtube:: 4XsTXt_9_SY
   :width: 120%

Endpointing Demo (Chinese)
--------------------------

The following two videos show endpointing demos using a pretrained model
on the `WenetSpeech`_ dataset. You can find the usage in
:ref:`lstm_server_chinese`.

Short Demo
^^^^^^^^^^

..  youtube:: sRQPGMZFun4
   :width: 120%

Long Demo
^^^^^^^^^

..  youtube:: LJtPJmX5jpE
   :width: 120%

Endpointing Demo (Arabic)
-------------------------

..  youtube:: t2SlrzgMd_k
   :width: 120%
