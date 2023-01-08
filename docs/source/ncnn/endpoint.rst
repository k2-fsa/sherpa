Endpointing
===========

We have three rules for endpoint detection. If any of them is activated,
we assume an endpoint is detected.

.. note::

  We borrow the implementation from

  `<https://kaldi-asr.org/doc/structkaldi_1_1OnlineEndpointRule.html>`_

Rule 1
------

In ``Rule 1``, we count the duration of trailing silence. If it is larger than
a user specified value, ``Rule 1`` is activated. The following is an example,
which uses ``2.4 seconds`` as the threshold.

  .. figure:: ./pic/rule1.png
     :alt: Rule 1 for endpoint detection
     :width: 600

Two cases are given:

(1) In the first case, nothing has been decoded when the duration of trailing
    silence reaches 2.4 seconds.

(2) In the second case, we first decode something before the duration of
    trailing silence reaches 2.4 seconds.

In both cases, ``Rule 1`` is activated.

.. hint::

  In the Python API, you can specify ``rule1_min_trailing_silence`` while
  constructing an instance of ``sherpa_ncnn.Recognizer``.

  In the C++ API, you can specify ``rule1.min_trailing_silence`` when creating
  ``EndpointConfig``.


Rule 2
------

In ``Rule 2``, we require that it has to first decode something
before we count the trailing silence. In the following example, after decoding
something, ``Rule 2`` is activated when the duration of trailing silence is
larger than the user specified value ``1.2`` seconds.

  .. figure:: ./pic/rule2.png
     :alt: Rule 2 for endpoint detection
     :width: 600

.. hint::

  In the Python API, you can specify ``rule2_min_trailing_silence`` while
  constructing an instance of ``sherpa_ncnn.Recognizer``.

  In the C++ API, you can specify ``rule2.min_trailing_silence`` when creating
  ``EndpointConfig``.

Rule 3
------

``Rule 3`` is activated when the utterance length in seconds is larger than
a given value. In the following example, ``Rule 3`` is activated after the
first segment reaches a given value, which is ``20`` seconds in this case.

  .. figure:: ./pic/rule3.png
     :alt: Rule 3 for endpoint detection
     :width: 600

.. hint::

  In the Python API, you can specify ``rule3_min_utterance_length`` while
  constructing an instance of ``sherpa_ncnn.Recognizer``.

  In the C++ API, you can specify ``rule3.min_utterance_length`` when creating
  ``EndpointConfig``.

.. note::

  If you want to deactive this rule, please provide a very large value
  for ``rule3_min_utterance_length`` or ``rule3.min_utterance_length``.

Demo
----

Multilingual (Chinese + English)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following video demonstrates using the Python API of `sherpa-ncnn`_
for real-time speech recogntinion with endpointing.

.. raw:: html

  <iframe src="//player.bilibili.com/player.html?bvid=BV1eK411y788&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="800" height="600"> </iframe>


.. hint::

  The code is available at

  `<https://github.com/k2-fsa/sherpa-ncnn/blob/master/python-api-examples/speech-recognition-from-microphone-with-endpoint-detection.py>`_

FAQs
----

How to compute duration of silence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For each frame to be decoded, we can output either a blank or a non-blank token.
We record the number of contiguous blanks that has been decoded so far.
In the current default setting, each frame is ``10 ms``. Thus, we can get
the duration of trailing silence by counting the number of contiguous trailing
blanks.

.. note::

  If a model uses a subsampling factor of 4, the time resolution becomes
  ``10 * 4 = 40 ms``.
