.. _onnx-x-asr-pascal-api:

Pascal
======

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Pascal API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- The Free Pascal compiler ``fpc``
- The Pascal API of `sherpa-onnx`_ (``sherpa-onnx/pascal-api``) and the C API library. Please first build `sherpa-onnx`_ from source so that ``libsherpa-onnx-c-api`` is available

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

Save the following as ``x-asr-offline.pas``:

.. code-block:: pascal

   { Copyright (c)  2026  Xiaomi Corporation }

   {
   This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   in Pascal.

   Before you run it, please download the model:

     wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
     tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   }
   program xasr_offline;

   {$mode objfpc}

   uses
     sherpa_onnx,
     DateUtils,
     SysUtils;

   var
     Wave: TSherpaOnnxWave;
     WaveFilename: AnsiString;

     Config: TSherpaOnnxOfflineRecognizerConfig;
     Recognizer: TSherpaOnnxOfflineRecognizer;
     Stream: TSherpaOnnxOfflineStream;
     RecognitionResult: TSherpaOnnxOfflineRecognizerResult;

     Start: TDateTime;
     Stop: TDateTime;

     Elapsed: Single;
     Duration: Single;
     ModelDir: AnsiString;
   begin
     ModelDir := './sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03';

     { Step 1: Create the recognizer from the X-ASR model files. }
     Initialize(Config);

     Config.ModelConfig.Transducer.Encoder := ModelDir + '/encoder-epoch-99-avg-1.int8.onnx';
     Config.ModelConfig.Transducer.Decoder := ModelDir + '/decoder-epoch-99-avg-1.onnx';
     Config.ModelConfig.Transducer.Joiner := ModelDir + '/joiner-epoch-99-avg-1.int8.onnx';
     Config.ModelConfig.Tokens := ModelDir + '/tokens.txt';
     Config.ModelConfig.Provider := 'cpu';
     Config.ModelConfig.NumThreads := 2;
     Config.ModelConfig.Debug := False;

     { Step 2: Read the wave file you want to transcribe. }
     WaveFilename := ModelDir + '/test_wavs/0.wav';
     Wave := SherpaOnnxReadWave(WaveFilename);

     { Step 3: Create a stream, send the whole audio to it, and decode. }
     Recognizer := TSherpaOnnxOfflineRecognizer.Create(Config);
     Stream := Recognizer.CreateStream();
     Start := Now;

     Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);
     Recognizer.Decode(Stream);

     RecognitionResult := Recognizer.GetResult(Stream);

     Stop := Now;

     { Step 4: Print the result. X-ASR punct models also add punctuations. }
     WriteLn(WaveFilename);
     WriteLn(RecognitionResult.Text);

     Elapsed := MilliSecondsBetween(Stop, Start) / 1000;
     Duration := Length(Wave.Samples) / Wave.SampleRate;
     WriteLn(Format('audio duration: %.3f s', [Duration]));
     WriteLn(Format('elapsed seconds: %.3f s', [Elapsed]));
     WriteLn(Format('RTF = %.3f/%.3f = %.3f', [Elapsed, Duration, Elapsed / Duration]));

     FreeAndNil(Stream);
     FreeAndNil(Recognizer);
   end.

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
   └── x-asr-offline.pas

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   2.4K 19 Nov 12:41 x-asr-offline.pas

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   fpc \
     -dSHERPA_ONNX_USE_SHARED_LIBS \
     -Fu/path/to/sherpa-onnx/sherpa-onnx/pascal-api \
     -Fl/path/to/sherpa-onnx/build/install/lib \
     x-asr-offline.pas

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
   elapsed seconds: 0.113 s
   RTF = 0.113/10.053 = 0.011

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

Save the following as ``x-asr-streaming.pas``:

.. code-block:: pascal

   { Copyright (c)  2026  Xiaomi Corporation }

   {
   This file shows how to use a streaming X-ASR model with sherpa-onnx
   in Pascal.

   Before you run it, please download the model:

     wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
     tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   }
   program xasr_streaming;

   {$mode objfpc}

   uses
     sherpa_onnx,
     DateUtils,
     SysUtils;

   var
     Wave: TSherpaOnnxWave;
     WaveFilename: AnsiString;

     Config: TSherpaOnnxOnlineRecognizerConfig;
     Recognizer: TSherpaOnnxOnlineRecognizer;
     Stream: TSherpaOnnxOnlineStream;
     RecognitionResult: TSherpaOnnxOnlineRecognizerResult;

     Start: TDateTime;
     Stop: TDateTime;

     Elapsed: Single;
     Duration: Single;
     ModelDir: AnsiString;

     ChunkSize: Integer;
     Chunk: array of Single;
     Tail: array of Single;
     I: Integer;
     Last: AnsiString;
   begin
     ModelDir := './sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05';

     { Step 1: Create the recognizer from the X-ASR model files. }
     Initialize(Config);

     Config.ModelConfig.Transducer.Encoder := ModelDir + '/encoder.int8.onnx';
     Config.ModelConfig.Transducer.Decoder := ModelDir + '/decoder.onnx';
     Config.ModelConfig.Transducer.Joiner := ModelDir + '/joiner.int8.onnx';
     Config.ModelConfig.Tokens := ModelDir + '/tokens.txt';
     Config.ModelConfig.Provider := 'cpu';
     Config.ModelConfig.NumThreads := 2;
     Config.ModelConfig.Debug := False;

     { Step 2: Read the wave file you want to transcribe. }
     WaveFilename := ModelDir + '/test_wavs/0.wav';
     Wave := SherpaOnnxReadWave(WaveFilename);

     { Step 3: Create a stream and feed the audio in small chunks, just like
       a microphone would do in a real-time application. }
     Recognizer := TSherpaOnnxOnlineRecognizer.Create(Config);
     Stream := Recognizer.CreateStream();

     Start := Now;

     ChunkSize := 3200; { 0.2 second for 16 kHz audio }
     Last := '';
     I := 0;
     while I < Length(Wave.Samples) do
     begin
       SetLength(Chunk, ChunkSize);
       if I + ChunkSize > Length(Wave.Samples) then
         SetLength(Chunk, Length(Wave.Samples) - I);
       Move(Wave.Samples[I], Chunk[0], Length(Chunk) * SizeOf(Single));

       Stream.AcceptWaveform(Chunk, Wave.SampleRate);

       while Recognizer.IsReady(Stream) do
         Recognizer.Decode(Stream);

       { Print the latest partial result when it changes. }
       RecognitionResult := Recognizer.GetResult(Stream);
       if (RecognitionResult.Text <> '') and (RecognitionResult.Text <> Last) then
       begin
         Last := RecognitionResult.Text;
         WriteLn(Last);
       end;

       I := I + ChunkSize;
     end;

     { Send some tail padding so the last words are flushed out. }
     SetLength(Tail, Round(0.5 * Wave.SampleRate));
     Stream.AcceptWaveform(Tail, Wave.SampleRate);
     Stream.InputFinished;
     while Recognizer.IsReady(Stream) do
       Recognizer.Decode(Stream);

     { Step 4: Print the final result. }
     Stop := Now;
     RecognitionResult := Recognizer.GetResult(Stream);
     WriteLn('Final result: ' + RecognitionResult.Text);

     Elapsed := MilliSecondsBetween(Stop, Start) / 1000;
     Duration := Length(Wave.Samples) / Wave.SampleRate;
     WriteLn(Format('audio duration: %.3f s', [Duration]));
     WriteLn(Format('elapsed seconds: %.3f s', [Elapsed]));
     WriteLn(Format('RTF = %.3f/%.3f = %.3f', [Elapsed, Duration, Elapsed / Duration]));

     FreeAndNil(Stream);
     FreeAndNil(Recognizer);
   end.

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
   └── x-asr-streaming.pas

   3 directories, 12 files

   $ ls -lh
   total 8
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   3.4K 19 Nov 12:41 x-asr-streaming.pas

Step 5. Build and run
^^^^^^^^^^^^^^^^^^^^^

Compile the code (please adjust the paths to your sherpa-onnx build) and run it:

.. code-block:: bash

   fpc \
     -dSHERPA_ONNX_USE_SHARED_LIBS \
     -Fu/path/to/sherpa-onnx/sherpa-onnx/pascal-api \
     -Fl/path/to/sherpa-onnx/build/install/lib \
     x-asr-streaming.pas

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
   elapsed seconds: 0.313 s
   RTF = 0.313/10.053 = 0.031

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

