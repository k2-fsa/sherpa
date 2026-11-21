.. _onnx-x-asr-c-api:

C
=

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the C API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- A C compiler
- The C API of `sherpa-onnx`_. Please first build `sherpa-onnx`_ from source so that ``libsherpa-onnx-c-api`` and ``libonnxruntime`` are available

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

Save the following as ``x-asr-offline.c``:

.. code-block:: c

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in C.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   #include <stdio.h>
   #include <string.h>
   #include <sys/time.h>

   #include "sherpa-onnx/c-api/c-api.h"

   int main() {
     const char *model_dir =
         "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03";

     char encoder[512], decoder[512], joiner[512], tokens[512], filename[512];
     snprintf(encoder, sizeof(encoder), "%s/encoder-epoch-99-avg-1.int8.onnx",
              model_dir);
     snprintf(decoder, sizeof(decoder), "%s/decoder-epoch-99-avg-1.onnx",
              model_dir);
     snprintf(joiner, sizeof(joiner), "%s/joiner-epoch-99-avg-1.int8.onnx",
              model_dir);
     snprintf(tokens, sizeof(tokens), "%s/tokens.txt", model_dir);
     snprintf(filename, sizeof(filename), "%s/test_wavs/0.wav", model_dir);

     // Step 1: Create the recognizer from the X-ASR model files.
     SherpaOnnxOfflineRecognizerConfig config;
     memset(&config, 0, sizeof(config));
     config.model_config.transducer.encoder = encoder;
     config.model_config.transducer.decoder = decoder;
     config.model_config.transducer.joiner = joiner;
     config.model_config.tokens = tokens;
     config.model_config.num_threads = 2;
     config.model_config.debug = 0;
     config.decoding_method = "greedy_search";

     const SherpaOnnxOfflineRecognizer *recognizer =
         SherpaOnnxCreateOfflineRecognizer(&config);

     // Step 2: Read the wave file you want to transcribe.
     const SherpaOnnxWave *wave = SherpaOnnxReadWave(filename);

     // Step 3: Create a stream, send the whole audio to it, and decode.
     struct timeval start, stop;
     gettimeofday(&start, NULL);

     const SherpaOnnxOfflineStream *stream =
         SherpaOnnxCreateOfflineStream(recognizer);
     SherpaOnnxAcceptWaveformOffline(stream, wave->sample_rate, wave->samples,
                                     wave->num_samples);
     SherpaOnnxDecodeOfflineStream(recognizer, stream);

     gettimeofday(&stop, NULL);

     // Step 4: Print the result. X-ASR punct models also add punctuations.
     const SherpaOnnxOfflineRecognizerResult *result =
         SherpaOnnxGetOfflineStreamResult(stream);
     printf("%s\n%s\n", filename, result->text);
     SherpaOnnxDestroyOfflineRecognizerResult(result);

     double elapsed = (stop.tv_sec - start.tv_sec) +
                      (stop.tv_usec - start.tv_usec) / 1e6;
     double audio_duration = (double)wave->num_samples / wave->sample_rate;
     printf("audio duration: %.3f s\n", audio_duration);
     printf("elapsed seconds: %.3f s\n", elapsed);
     printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audio_duration,
            elapsed / audio_duration);

     SherpaOnnxFreeWave(wave);
     SherpaOnnxDestroyOfflineStream(stream);
     SherpaOnnxDestroyOfflineRecognizer(recognizer);
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
   └── x-asr-offline.c

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   2.9K 19 Nov 12:41 x-asr-offline.c

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   cc x-asr-offline.c \
     -I /path/to/sherpa-onnx \
     -L /path/to/sherpa-onnx/build/install/lib \
     -lsherpa-onnx-c-api -lonnxruntime \
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
   elapsed seconds: 0.123 s
   RTF = 0.123/10.053 = 0.012

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

Save the following as ``x-asr-streaming.c``:

.. code-block:: c

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in C.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   #include <stdio.h>
   #include <string.h>
   #include <sys/time.h>

   #include "sherpa-onnx/c-api/c-api.h"

   int main() {
     const char *model_dir =
         "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-"
         "int8-2026-06-05";

     char encoder[512], decoder[512], joiner[512], tokens[512], filename[512];
     snprintf(encoder, sizeof(encoder), "%s/encoder.int8.onnx", model_dir);
     snprintf(decoder, sizeof(decoder), "%s/decoder.onnx", model_dir);
     snprintf(joiner, sizeof(joiner), "%s/joiner.int8.onnx", model_dir);
     snprintf(tokens, sizeof(tokens), "%s/tokens.txt", model_dir);
     snprintf(filename, sizeof(filename), "%s/test_wavs/0.wav", model_dir);

     // Step 1: Create the recognizer from the X-ASR model files.
     SherpaOnnxOnlineRecognizerConfig config;
     memset(&config, 0, sizeof(config));
     config.model_config.transducer.encoder = encoder;
     config.model_config.transducer.decoder = decoder;
     config.model_config.transducer.joiner = joiner;
     config.model_config.tokens = tokens;
     config.model_config.num_threads = 2;
     config.model_config.debug = 0;
     config.decoding_method = "greedy_search";

     const SherpaOnnxOnlineRecognizer *recognizer =
         SherpaOnnxCreateOnlineRecognizer(&config);

     // Step 2: Read the wave file you want to transcribe.
     const SherpaOnnxWave *wave = SherpaOnnxReadWave(filename);

     // Step 3: Feed the audio in small chunks, just like a microphone would do
     // in a real-time application.
     struct timeval start, stop;
     gettimeofday(&start, NULL);

     const SherpaOnnxOnlineStream *stream =
         SherpaOnnxCreateOnlineStream(recognizer);

     int32_t chunk = 3200;  // 0.2 second for 16 kHz audio
     char last[4096];
     last[0] = 0;

     for (int32_t i = 0; i < wave->num_samples; i += chunk) {
       int32_t end = i + chunk;
       if (end > wave->num_samples) {
         end = wave->num_samples;
       }
       SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate,
                                            wave->samples + i, end - i);

       while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
         SherpaOnnxDecodeOnlineStream(recognizer, stream);
       }

       // Print the latest partial result when it changes.
       const SherpaOnnxOnlineRecognizerResult *r =
           SherpaOnnxGetOnlineStreamResult(recognizer, stream);
       if (strlen(r->text) > 0 && strcmp(r->text, last) != 0) {
         strncpy(last, r->text, sizeof(last) - 1);
         last[sizeof(last) - 1] = 0;
         printf("%s\n", r->text);
       }
       SherpaOnnxDestroyOnlineRecognizerResult(r);
     }

     // Send some tail padding so the last words are flushed out.
     float tail[8000] = {0};  // 0.5 second at 16 kHz
     SherpaOnnxOnlineStreamAcceptWaveform(stream, wave->sample_rate, tail, 8000);
     SherpaOnnxOnlineStreamInputFinished(stream);
     while (SherpaOnnxIsOnlineStreamReady(recognizer, stream)) {
       SherpaOnnxDecodeOnlineStream(recognizer, stream);
     }

     // Step 4: Print the final result.
     gettimeofday(&stop, NULL);
     const SherpaOnnxOnlineRecognizerResult *r =
         SherpaOnnxGetOnlineStreamResult(recognizer, stream);
     printf("Final result: %s\n", r->text);
     SherpaOnnxDestroyOnlineRecognizerResult(r);

     double elapsed = (stop.tv_sec - start.tv_sec) +
                      (stop.tv_usec - start.tv_usec) / 1e6;
     double audio_duration = (double)wave->num_samples / wave->sample_rate;
     printf("audio duration: %.3f s\n", audio_duration);
     printf("elapsed seconds: %.3f s\n", elapsed);
     printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audio_duration,
            elapsed / audio_duration);

     SherpaOnnxFreeWave(wave);
     SherpaOnnxDestroyOnlineStream(stream);
     SherpaOnnxDestroyOnlineRecognizer(recognizer);
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
   └── x-asr-streaming.c

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   4.0K 19 Nov 12:41 x-asr-streaming.c

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   cc x-asr-streaming.c \
     -I /path/to/sherpa-onnx \
     -L /path/to/sherpa-onnx/build/install/lib \
     -lsherpa-onnx-c-api -lonnxruntime \
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
   elapsed seconds: 0.295 s
   RTF = 0.295/10.053 = 0.029

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

