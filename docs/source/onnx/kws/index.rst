.. _sherpa-onnx-keyword-spotting:

KWS Open vocabulary keyword spotting (Customized keyword spotting)
==================================================================

In this section, we describe how we implement the open vocabulary keyword spotting (aka customized keyword spotting)
feature and how to use it in `sherpa-onnx`_.

What is open vocabulary keyword spotting
----------------------------------------

Basically, an open vocabulary keyword spotting system is just like a tiny ASR system, but it can only decode words/phrases
in the given keywords. For example, if the given keyword is ``HELLO WORLD``, then the decoded result should be either
``HELLO WORLD`` or empty. As for open vocabulary (or customized), it means you can specify any keywords without re-training
the model. For building a conventional keyword spotting systems, people need to prepare a lot of audio-text pairs for the selected keywords
and the trained model can only be used to detect those selected keywords.
While an open vocabulary keyword spotting system allows people using one system to detect different keywords, even the keywords
might not be in the training data.


Decoder for open vocabulary keyword spotting
--------------------------------------------

For now, we only implement a beam search decoder to make the system only trigger the given keywords (i.e. the model itself is actually a tiny ASR).
To make it is able to balance between the ``trigged rate`` and ``false alarm``, we introduce two parameters for each keyword, ``boosting score``
and ``trigger threshold``.  The ``boosting score`` works like the hotwords recognition, it help the paths containing keywords to survive beam
search, the larger this score is the easier the corresponding keyword will be triggered, read :ref:`sherpa-onnx-hotwords` for more details.
The ``trigger threshold`` defines the minimum acoustic probability of decoded sequences (token sequences) that can be triggered, it is a float
value between 0 to 1, the lower this threshold is the easier the corresponding keyword will be triggered.

Keywords file
-------------

The input keywords looks like (the keywords are ``HELLO WORLD``, ``HI GOOGLE`` and ``HEY SIRI``):

.. code-block::

   ▁HE LL O ▁WORLD :1.5 #0.35
   ▁HI ▁GO O G LE :1.0 #0.25
   ▁HE Y ▁S I RI

Each line contains a keyword, the first several tokens (separated by spaces) are encoded tokens of the keyword, the item starts with ``:`` is the ``boosting score`` and the item starts with ``#`` is the ``trigger threshold``. Note: No spaces between ``:`` (or ``#``) and the float value.

To get the tokens you need to use the command line tool in `sherpa-onnx`_ to convert the original keywords, you can see the
usage as follows:

.. code-block::

   sherpa-onnx-cli text2token --help
   Usage: sherpa-onnx-cli text2token [OPTIONS] INPUT OUTPUT

   Options:

     --text TEXT         Path to the input texts. Each line in the texts contains the original phrase, it might also contain some extra items,
                         for example, the boosting score (startting with :), the triggering threshold
                         (startting with #, only used in keyword spotting task) and the original phrase (startting with @).
                         Note: extra items will be kept in the output.

                         example input 1 (tokens_type = ppinyin):
                             小爱同学 :2.0 #0.6 @小爱同学
                             你好问问 :3.5 @你好问问
                             小艺小艺 #0.6 @小艺小艺
                         example output 1:
                             x iǎo ài tóng x ué :2.0 #0.6 @小爱同学
                             n ǐ h ǎo w èn w èn :3.5 @你好问问
                             x iǎo y ì x iǎo y ì #0.6 @小艺小艺

                         example input 2 (tokens_type = bpe):
                             HELLO WORLD :1.5 #0.4
                             HI GOOGLE :2.0 #0.8
                             HEY SIRI #0.35
                         example output 2:
                             ▁HE LL O ▁WORLD :1.5 #0.4
                             ▁HI ▁GO O G LE :2.0 #0.8
                             ▁HE Y ▁S I RI #0.35

     --tokens TEXT       The path to tokens.txt.
     --tokens-type TEXT  The type of modeling units, should be cjkchar, bpe, cjkchar+bpe, fpinyin or ppinyin.
                         fpinyin means full pinyin, each cjkchar has a pinyin(with tone). ppinyin
                         means partial pinyin, it splits pinyin into initial and final,
     --bpe-model TEXT    The path to bpe.model. Only required when tokens-type is bpe or cjkchar+bpe.
     --help              Show this message and exit.

.. note::

   If the tokens-type is ``fpinyin`` or ``ppinyin``, you MUST provide the original keyword (starting with ``@``).

.. note::

   If you install sherpa-onnx from sources (i.e. not by pip), you can use the
   alternative script in `scripts`, the usage is almost the same as the command
   line tool, read the help information by:

   .. code-block::

     python3 scripts/text2token.py --help


How to use keyword spotting in sherpa-onnx
------------------------------------------

Currently, we provide command-line tool and android app for keyword spotting.


command-line tool
~~~~~~~~~~~~~~~~~

After installing `sherpa-onnx`_, type ``sherpa-onnx-keyword-spotter --help`` for the help message.

You can find the pre-trained models in :ref:`sherpa-onnx-kws-pre-trained-models`.


Android application
~~~~~~~~~~~~~~~~~~~

You can build your own application by starting with the ``build-kws-apk.sh`` in `sherpa-onnx`_ repository or referring to :ref:`sherpa-onnx-android`,
you can also try our generated apks from github release page.

Here is a demo video (Note: It is in Chinese).

.. raw:: html

   <iframe src="//player.bilibili.com/player.html?aid=326175636&bvid=BV1Nw411J7K6&cid=1405110216&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>




