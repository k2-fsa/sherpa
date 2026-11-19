.. _onnx-x-asr-cxx-api:

C++
===

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the C++ API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model. If you use the C API, the steps are the same; please see the header file ``sherpa-onnx/c-api/c-api.h``.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- A C++17 compiler
- The C++ API of `sherpa-onnx`_. Please first build `sherpa-onnx`_ from source so that ``libsherpa-onnx-cxx-api``, ``libsherpa-onnx-c-api``, and ``libonnxruntime`` are available

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

Save the following as ``x-asr-offline.cc``:

.. code-block:: cpp

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in C++.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   #include <chrono>
   #include <cstdio>
   #include <string>

   #include "sherpa-onnx/c-api/cxx-api.h"

   int main() {
     using namespace sherpa_onnx::cxx;  // NOLINT

     std::string model_dir =
         "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03";

     // Step 1: Create the recognizer from the X-ASR model files.
     OfflineRecognizerConfig config;
     config.model_config.transducer.encoder =
         model_dir + "/encoder-epoch-99-avg-1.int8.onnx";
     config.model_config.transducer.decoder =
         model_dir + "/decoder-epoch-99-avg-1.onnx";
     config.model_config.transducer.joiner =
         model_dir + "/joiner-epoch-99-avg-1.int8.onnx";
     config.model_config.tokens = model_dir + "/tokens.txt";
     config.model_config.num_threads = 2;
     config.model_config.debug = false;
     config.decoding_method = "greedy_search";

     auto recognizer = OfflineRecognizer::Create(config);

     // Step 2: Read the wave file you want to transcribe.
     std::string filename = model_dir + "/test_wavs/0.wav";
     auto wave = ReadWave(filename);

     // Step 3: Create a stream, send the whole audio to it, and decode.
     auto start = std::chrono::steady_clock::now();
     auto stream = recognizer.CreateStream();
     stream.AcceptWaveform(wave.sample_rate, wave.samples.data(),
                           wave.samples.size());
     recognizer.Decode(&stream);

     // Step 4: Print the result. X-ASR punct models also add punctuations.
     auto result = recognizer.GetResult(&stream);
     double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
     printf("%s\n%s\n", filename.c_str(), result.text.c_str());
     float audio_duration = wave.samples.size() / (float)wave.sample_rate;
     printf("audio duration: %.3f s\n", audio_duration);
     printf("elapsed seconds: %.3f s\n", elapsed);
     printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audio_duration, elapsed / audio_duration);

     return 0;
   }

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
   └── x-asr-offline.cc

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   2.3K 19 Nov 12:23 x-asr-offline.cc

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   c++ -std=c++17 x-asr-offline.cc \
     -I /path/to/sherpa-onnx \
     -L /path/to/sherpa-onnx/build/install/lib \
     -lsherpa-onnx-cxx-api -lsherpa-onnx-c-api -lonnxruntime \
     -o x-asr-offline

   # On macOS:
   export DYLD_LIBRARY_PATH=/path/to/sherpa-onnx/build/install/lib
   # On Linux:
   # export LD_LIBRARY_PATH=/path/to/sherpa-onnx/build/install/lib

   ./x-asr-offline

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.114 s
   RTF = 0.114/10.053 = 0.011

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

Save the following as ``x-asr-streaming.cc``:

.. code-block:: cpp

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in C++.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   #include <chrono>
   #include <cstdio>
   #include <string>
   #include <vector>

   #include "sherpa-onnx/c-api/cxx-api.h"

   int main() {
     using namespace sherpa_onnx::cxx;  // NOLINT

     std::string model_dir =
         "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-"
         "int8-2026-06-05";

     // Step 1: Create the recognizer from the X-ASR model files.
     OnlineRecognizerConfig config;
     config.model_config.transducer.encoder = model_dir + "/encoder.int8.onnx";
     config.model_config.transducer.decoder = model_dir + "/decoder.onnx";
     config.model_config.transducer.joiner = model_dir + "/joiner.int8.onnx";
     config.model_config.tokens = model_dir + "/tokens.txt";
     config.model_config.num_threads = 2;
     config.model_config.debug = false;
     config.decoding_method = "greedy_search";

     auto recognizer = OnlineRecognizer::Create(config);

     // Step 2: Read the wave file you want to transcribe.
     std::string filename = model_dir + "/test_wavs/0.wav";
     auto wave = ReadWave(filename);

     // Step 3: Feed the audio in small chunks, just like a microphone would do
     // in a real-time application.
     auto start = std::chrono::steady_clock::now();
     auto stream = recognizer.CreateStream();

     // 0.2 second for 16 kHz audio
     int32_t chunk = 3200;

     std::string last;
     for (int32_t i = 0; i < static_cast<int32_t>(wave.samples.size());
          i += chunk) {
       int32_t end = std::min(i + chunk, (int32_t)wave.samples.size());
       stream.AcceptWaveform(wave.sample_rate, wave.samples.data() + i, end - i);

       while (recognizer.IsReady(&stream)) {
         recognizer.Decode(&stream);
       }

       // Print the latest partial result when it changes.
       auto result = recognizer.GetResult(&stream);
       if (!result.text.empty() && result.text != last) {
         last = result.text;
         printf("%s\n", result.text.c_str());
       }
     }

     // Send some tail padding so the last words are flushed out.
     std::vector<float> tail(static_cast<int32_t>(0.5 * wave.sample_rate), 0.0f);
     stream.AcceptWaveform(wave.sample_rate, tail.data(), tail.size());
     stream.InputFinished();
     while (recognizer.IsReady(&stream)) {
       recognizer.Decode(&stream);
     }

     // Step 4: Print the final result.
     auto result = recognizer.GetResult(&stream);
     double elapsed = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
     printf("Final result: %s\n", result.text.c_str());
     float audio_duration = wave.samples.size() / (float)wave.sample_rate;
     printf("audio duration: %.3f s\n", audio_duration);
     printf("elapsed seconds: %.3f s\n", elapsed);
     printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audio_duration, elapsed / audio_duration);

     return 0;
   }

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
   └── x-asr-streaming.cc

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   3.1K 19 Nov 12:23 x-asr-streaming.cc

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   c++ -std=c++17 x-asr-streaming.cc \
     -I /path/to/sherpa-onnx \
     -L /path/to/sherpa-onnx/build/install/lib \
     -lsherpa-onnx-cxx-api -lsherpa-onnx-c-api -lonnxruntime \
     -o x-asr-streaming

   ./x-asr-streaming

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
   elapsed seconds: 0.309 s
   RTF = 0.309/10.053 = 0.031

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

