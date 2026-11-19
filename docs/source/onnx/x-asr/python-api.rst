.. _onnx-x-asr-python-api:

Python
======

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with `sherpa-onnx`_ in Python. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Python 3.8 or above
- `sherpa-onnx`_, installed with ``pip install sherpa-onnx``
- ``soundfile``: ``pip install soundfile``
- ``numpy``, which is installed together with `sherpa-onnx`_

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 3. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``x-asr-offline.py``:

.. code-block:: python

   #!/usr/bin/env python3
   # This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   # in Python.
   #
   # Before you run it, please download the model:
   #
   #   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   #   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   #
   # and install sherpa-onnx:
   #
   #   pip install sherpa-onnx
   import time

   import numpy as np
   import soundfile as sf

   import sherpa_onnx


   def read_wave(filename: str):
       """Read a WAVE file.

       Returns:
         A tuple of (samples, sample_rate), where samples is a 1-D float32
         array normalized to the range [-1, 1].
       """
       # See https://python-soundfile.readthedocs.io/en/0.11.0/#soundfile.read
       data, sample_rate = sf.read(
           filename,
           always_2d=True,
           dtype="float32",
       )
       data = data[:, 0]  # use only the first channel
       samples = np.ascontiguousarray(data)

       # samples is a 1-D array of dtype float32
       # sample_rate is a scalar
       return samples, sample_rate


   def main():
       model_dir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"

       # Step 1: Create the recognizer from the X-ASR model files.
       recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
           encoder=f"{model_dir}/encoder-epoch-99-avg-1.int8.onnx",
           decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
           joiner=f"{model_dir}/joiner-epoch-99-avg-1.int8.onnx",
           tokens=f"{model_dir}/tokens.txt",
           num_threads=2,
           debug=False,
       )

       # Step 2: Read the wave file you want to transcribe.
       filename = f"{model_dir}/test_wavs/0.wav"
       samples, sample_rate = read_wave(filename)

       # Step 3: Create a stream, send the whole audio to it, and decode.
       start = time.time()
       stream = recognizer.create_stream()
       stream.accept_waveform(sample_rate, samples)
       recognizer.decode_stream(stream)

       elapsed = time.time() - start

       # Step 4: Print the result. X-ASR punct models also add punctuations.
       print(f"{filename}\n{stream.result.text}")

       audio_duration = len(samples) / sample_rate
       print(f"audio duration: {audio_duration:.3f} s")
       print(f"elapsed seconds: {elapsed:.3f} s")
       print(f"RTF = {elapsed:.3f}/{audio_duration:.3f} = {elapsed / audio_duration:.3f}")


   if __name__ == "__main__":
       main()

Step 4. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   │   ├── bpe.model
   │   ├── decoder-epoch-99-avg-1.onnx
   │   ├── encoder-epoch-99-avg-1.int8.onnx
   │   ├── joiner-epoch-99-avg-1.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── x-asr-offline.py

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   2.4K 19 Nov 12:23 x-asr-offline.py

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 ./x-asr-offline.py

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.129 s
   RTF = 0.129/10.053 = 0.013

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Streaming X-ASR
---------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 3. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``x-asr-streaming.py``:

.. code-block:: python

   #!/usr/bin/env python3
   # This file shows how to use a streaming X-ASR model with sherpa-onnx
   # in Python.
   #
   # Before you run it, please download the model:
   #
   #   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   #   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   #
   # and install sherpa-onnx:
   #
   #   pip install sherpa-onnx
   import time

   import numpy as np
   import soundfile as sf

   import sherpa_onnx


   def read_wave(filename: str):
       """Read a WAVE file.

       Returns:
         A tuple of (samples, sample_rate), where samples is a 1-D float32
         array normalized to the range [-1, 1].
       """
       # See https://python-soundfile.readthedocs.io/en/0.11.0/#soundfile.read
       data, sample_rate = sf.read(
           filename,
           always_2d=True,
           dtype="float32",
       )
       data = data[:, 0]  # use only the first channel
       samples = np.ascontiguousarray(data)

       # samples is a 1-D array of dtype float32
       # sample_rate is a scalar
       return samples, sample_rate


   def main():
       model_dir = (
           "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"
       )

       # Step 1: Create the recognizer from the X-ASR model files.
       recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
           encoder=f"{model_dir}/encoder.int8.onnx",
           decoder=f"{model_dir}/decoder.onnx",
           joiner=f"{model_dir}/joiner.int8.onnx",
           tokens=f"{model_dir}/tokens.txt",
           num_threads=2,
           debug=False,
       )

       # Step 2: Read the wave file you want to transcribe.
       filename = f"{model_dir}/test_wavs/0.wav"
       samples, sample_rate = read_wave(filename)

       # Step 3: Create a stream and feed the audio in small chunks, just like
       # a microphone would do in a real-time application.
       start = time.time()
       stream = recognizer.create_stream()

       # 0.2 second for 16 kHz audio
       chunk_size = int(0.2 * sample_rate)

       last = ""
       for i in range(0, len(samples), chunk_size):
           stream.accept_waveform(sample_rate, samples[i : i + chunk_size])

           while recognizer.is_ready(stream):
               recognizer.decode_stream(stream)

           # Print the latest partial result when it changes.
           result = recognizer.get_result(stream)
           if result and result != last:
               last = result
               print(result)

       # Send some tail padding so the last words are flushed out.
       stream.accept_waveform(sample_rate, np.zeros(int(0.5 * sample_rate), dtype=np.float32))
       while recognizer.is_ready(stream):
           recognizer.decode_stream(stream)

       # Step 4: Print the final result.
       elapsed = time.time() - start
       print("Final result:", recognizer.get_result(stream))

       audio_duration = len(samples) / sample_rate
       print(f"audio duration: {audio_duration:.3f} s")
       print(f"elapsed seconds: {elapsed:.3f} s")
       print(f"RTF = {elapsed:.3f}/{audio_duration:.3f} = {elapsed / audio_duration:.3f}")


   if __name__ == "__main__":
       main()

Step 4. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   │   ├── bpe.model
   │   ├── decoder.onnx
   │   ├── encoder.int8.onnx
   │   ├── joiner.int8.onnx
   │   ├── README.md
   │   ├── test_onnx.py
   │   ├── test_wavs
   │   │   ├── 0.wav
   │   │   ├── 1.wav
   │   │   ├── 2.wav
   │   │   └── 3.wav
   │   └── tokens.txt
   └── x-asr-streaming.py

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   3.1K 19 Nov 12:23 x-asr-streaming.py

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 ./x-asr-streaming.py

Output
^^^^^^

.. code-block:: text

   昨
   昨天是
   昨天是 Mon
   昨天是 Monday
   昨天是 Monday，
   昨天是 Monday， today
   昨天是 Monday， today is 礼
   昨天是 Monday， today is 礼拜二
   昨天是 Monday， today is 礼拜二，
   昨天是 Monday， today is 礼拜二， the day
   昨天是 Monday， today is 礼拜二， the day after
   昨天是 Monday， today is 礼拜二， the day after tomorro
   昨天是 Monday， today is 礼拜二， the day after tomorrow
   昨天是 Monday， today is 礼拜二， the day after tomorrow 是
   Final result: 昨天是 Monday， today is 礼拜二， the day after tomorrow 是星期
   audio duration: 10.053 s
   elapsed seconds: 0.305 s
   RTF = 0.305/10.053 = 0.030

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

