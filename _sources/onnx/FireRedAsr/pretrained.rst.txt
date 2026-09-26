Pre-trained Models
==================

This page describes how to download pre-trained `FireRedAsr`_ models.

Note that we support models from the following two repositories

  - ``v1``: `<https://github.com/FireRedTeam/FireRedASR>`_
  - ``v2``: `<https://github.com/FireRedTeam/FireRedASR2S>`_

``v1`` contains only one model, based on attention-encoder-decoder and it is somewhat ``slow`` on CPU.

``v2`` contains one more CTC model, which is very ``fast`` on CPU. The AED model in ``v2`` is also much faster than the ``v1`` AED model.

.. include:: ./v2-ctc.rst
.. include:: ./v2-aed.rst
.. include:: ./v1-aed.rst


