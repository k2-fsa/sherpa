.. _try sherpa with huggingface:

Try sherpa with Huggingface
===========================

This page describes how to use `sherpa`_ for automatic speech recognition
with `Huggingface`_.

.. hint::

  You don't need to download or install anything. All you need is a browser.


The server is running on CPU within a docker container provided by
`Huggingface`_ and you use a browser to interact with it. The browser
can be run on Windows, macOS, Linux, or even on your phone or iPad.

You can either upload a file for recognition or record your speech via
a microphone from within the browser and submit it for recognition.

Now let's get started.

Visit our Huggingface space
---------------------------

Start your browser and visit the following address:

`<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_

and you will see a page like the following screenshot:

.. image:: ./pic/hugging-face-sherpa.png
   :alt: screenshot of `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_
   :target: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition

You can:

  1. Select a language for recognition. Currently, we provide pre-trained models
     from `icefall`_ for the following languages: ``Chinese``, ``English``, and
     ``Chinese+English``.
  2. After selecting the target language, you can select a pre-trained model
     corresponding to the language.
  3. Select the decoding method. Currently, it provides ``greedy search``
     and ``modified_beam_search``.
  4. If you selected ``modified_beam_search``, you can choose the number of
     active paths during the search.
  5. Either upload a file or record your speech for recognition.
  6. Click the button ``Submit for recognition``.
  7. Wait for a moment and you will get the recognition results.

The following screenshot shows an example when selecting ``Chinese+English``:

.. image:: ./pic/hugging-face-sherpa-3.png
   :alt: screenshot of `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_
   :target: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition


In the bottom part of the page, you can find a table of examples. You can click
one of them and then click ``Submit for recognition``.

.. image:: ./pic/hugging-face-sherpa-2.png
   :alt: screenshot of `<https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition>`_
   :target: https://huggingface.co/spaces/k2-fsa/automatic-speech-recognition