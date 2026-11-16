.. _sherpa-onnx-forced-alignment:

Forced alignment
================

Forced alignment computes the start and end time of every word of an audio
file given its transcript. For instance, for the audio

.. code-block:: text

   李雷和韩梅梅是好朋友

we can obtain that ``李`` is from 0.16 s to 0.38 s, ``雷`` is from 0.38 s to
0.62 s, and so on.

`sherpa-onnx`_ supports forced alignment with GMM acoustic models trained
with the `Montreal Forced Aligner`_ (MFA). The models and their pronunciation
lexicons are packaged in the `fa-models`_ release; see
`scripts/mfa/README.md <https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/mfa/README.md>`_
for what is inside a package and where the models come from.

.. hint::

   It runs locally on your device, without internet connection. No audio is
   uploaded anywhere.

The alignment runs with the same `sherpa-onnx`_ APIs as the other functions,
in every supported language (C, C++, Python, Java, Kotlin, JavaScript, ...).
Given a wave and its transcript, the result contains a phone tier and a word
tier with the time interval of each phone/word; it can also be written as a
`Praat <https://www.fon.hum.uva.nl/praat/>`_ TextGrid file.

In the following, we describe how to use pre-built apps for forced alignment
on Android and in a web browser.

.. toctree::
   :maxdepth: 2

   ./android.rst
   ./web.rst
