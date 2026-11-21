.. _onnx-x-asr-dart-api:

Dart/Flutter
============

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with `sherpa-onnx`_ in Dart and Flutter. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Dart 3.0 or above (or Flutter)

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``pubspec.yaml``:

.. code-block:: yaml

   name: x_asr_demo
   environment:
     sdk: '>=3.0.0 <4.0.0'
   dependencies:
     sherpa_onnx: ^2.0.1


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``bin/x-asr-offline.dart``:

.. code-block:: dart

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in Dart/Flutter.
   //
   // Before you run it, please add sherpa_onnx to your pubspec.yaml:
   //
   //   dependencies:
   //     sherpa_onnx: ^2.0.1
   //
   // and download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

   Future<void> main() async {
     // Load the sherpa-onnx native library.
     await sherpa_onnx.initBindingsAsync();

     const modelDir =
         './sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03';

     // Step 1: Create the recognizer from the X-ASR model files.
     final transducer = sherpa_onnx.OfflineTransducerModelConfig(
       encoder: '$modelDir/encoder-epoch-99-avg-1.int8.onnx',
       decoder: '$modelDir/decoder-epoch-99-avg-1.onnx',
       joiner: '$modelDir/joiner-epoch-99-avg-1.int8.onnx',
     );
     final modelConfig = sherpa_onnx.OfflineModelConfig(
       transducer: transducer,
       tokens: '$modelDir/tokens.txt',
       numThreads: 2,
       debug: false,
     );
     final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
     final recognizer = sherpa_onnx.OfflineRecognizer(config);

     // Step 2: Read the wave file you want to transcribe.
     final filename = '$modelDir/test_wavs/0.wav';
     final wave = sherpa_onnx.readWave(filename);

     // Step 3: Create a stream, send the whole audio to it, and decode.
     final stopwatch = Stopwatch()..start();
     final stream = recognizer.createStream();
     stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
     recognizer.decode(stream);

     // Step 4: Print the result. X-ASR punct models also add punctuations.
     print(filename);
     stopwatch.stop();
     final elapsed = stopwatch.elapsedMicroseconds / 1e6;
     print(recognizer.getResult(stream).text);

     final audioDuration = wave.samples.length / wave.sampleRate;
     print('audio duration: ${audioDuration.toStringAsFixed(3)} s');
     print('elapsed seconds: ${elapsed.toStringAsFixed(3)} s');
     print('RTF = ${elapsed.toStringAsFixed(3)}/${audioDuration.toStringAsFixed(3)} = ${(elapsed / audioDuration).toStringAsFixed(3)}');

     stream.free();
     recognizer.free();
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── bin
   │   └── x-asr-offline.dart
   ├── pubspec.yaml
   └── sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
       ├── bpe.model
       ├── decoder-epoch-99-avg-1.onnx
       ├── encoder-epoch-99-avg-1.int8.onnx
       ├── joiner-epoch-99-avg-1.int8.onnx
       ├── README.md
       ├── test_onnx.py
       ├── test_wavs
       │   ├── 0.wav
       │   ├── 1.wav
       │   ├── 2.wav
       │   └── 3.wav
       └── tokens.txt

   4 directories, 13 files

   $ ls -lh
   total 8
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 bin
   -rw-r--r--@  1 fangjun  wheel    90B 19 Nov 12:23 pubspec.yaml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dart pub get
   dart run ./bin/x-asr-offline.dart

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.200 s
   RTF = 0.200/10.053 = 0.020

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Streaming X-ASR
---------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``pubspec.yaml``:

.. code-block:: yaml

   name: x_asr_demo
   environment:
     sdk: '>=3.0.0 <4.0.0'
   dependencies:
     sherpa_onnx: ^2.0.1


Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``bin/x-asr-streaming.dart``:

.. code-block:: dart

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in Dart/Flutter.
   //
   // Before you run it, please add sherpa_onnx to your pubspec.yaml:
   //
   //   dependencies:
   //     sherpa_onnx: ^2.0.1
   //
   // and download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   import 'dart:typed_data';

   import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

   Future<void> main() async {
     // Load the sherpa-onnx native library.
     await sherpa_onnx.initBindingsAsync();

     const modelDir =
         './sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05';

     // Step 1: Create the recognizer from the X-ASR model files.
     final transducer = sherpa_onnx.OnlineTransducerModelConfig(
       encoder: '$modelDir/encoder.int8.onnx',
       decoder: '$modelDir/decoder.onnx',
       joiner: '$modelDir/joiner.int8.onnx',
     );
     final modelConfig = sherpa_onnx.OnlineModelConfig(
       transducer: transducer,
       tokens: '$modelDir/tokens.txt',
       numThreads: 2,
       debug: false,
     );
     final config = sherpa_onnx.OnlineRecognizerConfig(model: modelConfig);
     final recognizer = sherpa_onnx.OnlineRecognizer(config);

     // Step 2: Read the wave file you want to transcribe.
     final filename = '$modelDir/test_wavs/0.wav';
     final wave = sherpa_onnx.readWave(filename);

     // Step 3: Create a stream and feed the audio in small chunks, just like
     // a microphone would do in a real-time application.
     final stopwatch = Stopwatch()..start();
     final stream = recognizer.createStream();

     // 0.2 second for 16 kHz audio
     final chunkSize = 3200;
     final numChunks = wave.samples.length ~/ chunkSize;

     var last = '';
     for (int i = 0; i < numChunks; ++i) {
       int start = i * chunkSize;
       stream.acceptWaveform(
         samples: Float32List.sublistView(
             wave.samples, start, start + chunkSize),
         sampleRate: wave.sampleRate,
       );

       while (recognizer.isReady(stream)) {
         recognizer.decode(stream);
       }

       // Print the latest partial result when it changes.
       final text = recognizer.getResult(stream).text;
       if (text.isNotEmpty && text != last) {
         last = text;
         print(text);
       }
     }

     // Send some tail padding so the last words are flushed out.
     stream.acceptWaveform(
       samples: Float32List((0.5 * wave.sampleRate).toInt()),
       sampleRate: wave.sampleRate,
     );
     while (recognizer.isReady(stream)) {
       recognizer.decode(stream);
     }

     // Step 4: Print the final result.
     stopwatch.stop();
     final elapsed = stopwatch.elapsedMicroseconds / 1e6;
     print('Final result: ${recognizer.getResult(stream).text}');

     final audioDuration = wave.samples.length / wave.sampleRate;
     print('audio duration: ${audioDuration.toStringAsFixed(3)} s');
     print('elapsed seconds: ${elapsed.toStringAsFixed(3)} s');
     print('RTF = ${elapsed.toStringAsFixed(3)}/${audioDuration.toStringAsFixed(3)} = ${(elapsed / audioDuration).toStringAsFixed(3)}');

     stream.free();
     recognizer.free();
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── bin
   │   └── x-asr-streaming.dart
   ├── pubspec.yaml
   └── sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
       ├── bpe.model
       ├── decoder.onnx
       ├── encoder.int8.onnx
       ├── joiner.int8.onnx
       ├── README.md
       ├── test_onnx.py
       ├── test_wavs
       │   ├── 0.wav
       │   ├── 1.wav
       │   ├── 2.wav
       │   └── 3.wav
       └── tokens.txt

   4 directories, 13 files

   $ ls -lh
   total 8
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 bin
   -rw-r--r--@  1 fangjun  wheel    90B 19 Nov 12:23 pubspec.yaml
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dart pub get
   dart run ./bin/x-asr-streaming.dart

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
   elapsed seconds: 0.356 s
   RTF = 0.356/10.053 = 0.035

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

In a Flutter app, call ``await sherpa_onnx.initBindingsAsync()`` once before using the API; the rest of the code is the same.

