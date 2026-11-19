.. _onnx-x-asr-csharp-api:

C#
==

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the C# API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- .NET 8.0 or above

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``x-asr-demo.csproj``:

.. code-block:: xml

   <Project Sdk="Microsoft.NET.Sdk">

     <PropertyGroup>
       <OutputType>Exe</OutputType>
       <TargetFramework>net8.0</TargetFramework>
       <ImplicitUsings>enable</ImplicitUsings>
       <Nullable>enable</Nullable>
     </PropertyGroup>

     <ItemGroup>
       <PackageReference Include="org.k2fsa.sherpa.onnx" Version="*" />
     </ItemGroup>

   </Project>

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Program.cs``:

.. code-block:: csharp

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in C#.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   using System.Diagnostics;
   using SherpaOnnx;

   public class XasrOfflineDemo
   {
     public static void Main(string[] args)
     {
       string modelDir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03";

       // Step 1: Create the recognizer from the X-ASR model files.
       var config = new OfflineRecognizerConfig();
       config.ModelConfig.Transducer.Encoder = modelDir + "/encoder-epoch-99-avg-1.int8.onnx";
       config.ModelConfig.Transducer.Decoder = modelDir + "/decoder-epoch-99-avg-1.onnx";
       config.ModelConfig.Transducer.Joiner = modelDir + "/joiner-epoch-99-avg-1.int8.onnx";
       config.ModelConfig.Tokens = modelDir + "/tokens.txt";
       config.ModelConfig.NumThreads = 2;
       config.ModelConfig.Debug = 0;
       config.DecodingMethod = "greedy_search";

       var recognizer = new OfflineRecognizer(config);

       // Step 2: Read the wave file you want to transcribe.
       string filename = modelDir + "/test_wavs/0.wav";
       var reader = new SimpleWaveReader(filename);

       // Step 3: Create a stream, send the whole audio to it, and decode.
       var stopwatch = Stopwatch.StartNew();
       var stream = recognizer.CreateStream();
       stream.AcceptWaveform(reader.SampleRate, reader.Samples);
       recognizer.Decode(stream);

       // Step 4: Print the result. X-ASR punct models also add punctuations.
       Console.WriteLine(filename);
       stopwatch.Stop();
       double elapsed = stopwatch.Elapsed.TotalSeconds;
       Console.WriteLine(stream.Result.Text);

       double audioDuration = (double)reader.Samples.Length / reader.SampleRate;
       Console.WriteLine($"audio duration: {audioDuration:F3} s");
       Console.WriteLine($"elapsed seconds: {elapsed:F3} s");
       Console.WriteLine($"RTF = {elapsed:F3}/{audioDuration:F3} = {elapsed / audioDuration:F3}");

       recognizer.Dispose();
     }
   }

   // A minimal WAVE file reader for mono, 16-bit wav files.
   class SimpleWaveReader
   {
     public float[] Samples { get; }
     public int SampleRate { get; }

     public SimpleWaveReader(string filename)
     {
       using var fs = File.OpenRead(filename);
       using var br = new BinaryReader(fs);
       if (new string(br.ReadChars(4)) != "RIFF") throw new Exception("Not a WAVE file");
       br.ReadInt32(); // file size
       if (new string(br.ReadChars(4)) != "WAVE") throw new Exception("Not a WAVE file");

       int sampleRate = 0;
       short bitsPerSample = 0;
       short numChannels = 0;
       byte[] data = Array.Empty<byte>();

       while (fs.Position < fs.Length)
       {
         string chunkId = new string(br.ReadChars(4));
         int chunkSize = br.ReadInt32();
         if (chunkId == "fmt ")
         {
           br.ReadInt16(); // audio format
           numChannels = br.ReadInt16();
           sampleRate = br.ReadInt32();
           br.ReadInt32(); // byte rate
           br.ReadInt16(); // block align
           bitsPerSample = br.ReadInt16();
         }
         else if (chunkId == "data")
         {
           data = br.ReadBytes(chunkSize);
         }
         else
         {
           fs.Seek(chunkSize, SeekOrigin.Current);
         }
       }

       if (numChannels != 1 || bitsPerSample != 16)
       {
         throw new Exception("Please use a mono, 16-bit WAVE file");
       }

       int numSamples = data.Length / 2;
       var samples = new float[numSamples];
       for (int i = 0; i < numSamples; i++)
       {
         short s = BitConverter.ToInt16(data, i * 2);
         samples[i] = s / 32768.0f;
       }
       Samples = samples;
       SampleRate = sampleRate;
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Program.cs
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
   └── x-asr-demo.csproj

   3 directories, 13 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel   3.7K 19 Nov 12:23 Program.cs
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   -rw-r--r--@  1 fangjun  wheel   338B 19 Nov 12:23 x-asr-demo.csproj

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dotnet run

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.121 s
   RTF = 0.121/10.053 = 0.012

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

Save the following as ``x-asr-demo.csproj``:

.. code-block:: xml

   <Project Sdk="Microsoft.NET.Sdk">

     <PropertyGroup>
       <OutputType>Exe</OutputType>
       <TargetFramework>net8.0</TargetFramework>
       <ImplicitUsings>enable</ImplicitUsings>
       <Nullable>enable</Nullable>
     </PropertyGroup>

     <ItemGroup>
       <PackageReference Include="org.k2fsa.sherpa.onnx" Version="*" />
     </ItemGroup>

   </Project>

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Program.cs``:

.. code-block:: csharp

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in C#.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   using System.Diagnostics;
   using SherpaOnnx;

   public class XasrStreamingDemo
   {
     public static void Main(string[] args)
     {
       string modelDir =
           "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05";

       // Step 1: Create the recognizer from the X-ASR model files.
       var config = new OnlineRecognizerConfig();
       config.ModelConfig.Transducer.Encoder = modelDir + "/encoder.int8.onnx";
       config.ModelConfig.Transducer.Decoder = modelDir + "/decoder.onnx";
       config.ModelConfig.Transducer.Joiner = modelDir + "/joiner.int8.onnx";
       config.ModelConfig.Tokens = modelDir + "/tokens.txt";
       config.ModelConfig.NumThreads = 2;
       config.ModelConfig.Debug = 0;
       config.DecodingMethod = "greedy_search";

       var recognizer = new OnlineRecognizer(config);

       // Step 2: Read the wave file you want to transcribe.
       string filename = modelDir + "/test_wavs/0.wav";
       var reader = new SimpleWaveReader(filename);
       float[] samples = reader.Samples;
       int sampleRate = reader.SampleRate;

       // Step 3: Create a stream and feed the audio in small chunks, just like
       // a microphone would do in a real-time application.
       var stopwatch = Stopwatch.StartNew();
       var stream = recognizer.CreateStream();

       // 0.2 second for 16 kHz audio
       int chunk = 3200;

       string last = "";
       for (int i = 0; i < samples.Length; i += chunk)
       {
         int end = Math.Min(i + chunk, samples.Length);
         float[] slice = new float[end - i];
         Array.Copy(samples, i, slice, 0, slice.Length);
         stream.AcceptWaveform(sampleRate, slice);

         while (recognizer.IsReady(stream))
         {
           recognizer.Decode(stream);
         }

         // Print the latest partial result when it changes.
         string text = recognizer.GetResult(stream).Text;
         if (text.Length > 0 && text != last)
         {
           last = text;
           Console.WriteLine(text);
         }
       }

       // Send some tail padding so the last words are flushed out.
       stream.AcceptWaveform(sampleRate, new float[(int)(0.5 * sampleRate)]);
       while (recognizer.IsReady(stream))
       {
         recognizer.Decode(stream);
       }

       // Step 4: Print the final result.
       stopwatch.Stop();
       double elapsed = stopwatch.Elapsed.TotalSeconds;
       Console.WriteLine("Final result: " + recognizer.GetResult(stream).Text);

       double audioDuration = (double)reader.Samples.Length / reader.SampleRate;
       Console.WriteLine($"audio duration: {audioDuration:F3} s");
       Console.WriteLine($"elapsed seconds: {elapsed:F3} s");
       Console.WriteLine($"RTF = {elapsed:F3}/{audioDuration:F3} = {elapsed / audioDuration:F3}");

       recognizer.Dispose();
     }
   }

   // A minimal WAVE file reader for mono, 16-bit wav files.
   class SimpleWaveReader
   {
     public float[] Samples { get; }
     public int SampleRate { get; }

     public SimpleWaveReader(string filename)
     {
       using var fs = File.OpenRead(filename);
       using var br = new BinaryReader(fs);
       if (new string(br.ReadChars(4)) != "RIFF") throw new Exception("Not a WAVE file");
       br.ReadInt32(); // file size
       if (new string(br.ReadChars(4)) != "WAVE") throw new Exception("Not a WAVE file");

       int sampleRate = 0;
       short bitsPerSample = 0;
       short numChannels = 0;
       byte[] data = Array.Empty<byte>();

       while (fs.Position < fs.Length)
       {
         string chunkId = new string(br.ReadChars(4));
         int chunkSize = br.ReadInt32();
         if (chunkId == "fmt ")
         {
           br.ReadInt16(); // audio format
           numChannels = br.ReadInt16();
           sampleRate = br.ReadInt32();
           br.ReadInt32(); // byte rate
           br.ReadInt16(); // block align
           bitsPerSample = br.ReadInt16();
         }
         else if (chunkId == "data")
         {
           data = br.ReadBytes(chunkSize);
         }
         else
         {
           fs.Seek(chunkSize, SeekOrigin.Current);
         }
       }

       if (numChannels != 1 || bitsPerSample != 16)
       {
         throw new Exception("Please use a mono, 16-bit WAVE file");
       }

       int numSamples = data.Length / 2;
       var samples = new float[numSamples];
       for (int i = 0; i < numSamples; i++)
       {
         short s = BitConverter.ToInt16(data, i * 2);
         samples[i] = s / 32768.0f;
       }
       Samples = samples;
       SampleRate = sampleRate;
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Program.cs
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
   └── x-asr-demo.csproj

   3 directories, 13 files

   $ ls -lh
   total 24
   -rw-r--r--@  1 fangjun  wheel   4.6K 19 Nov 12:23 Program.cs
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   -rw-r--r--@  1 fangjun  wheel   338B 19 Nov 12:23 x-asr-demo.csproj

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   dotnet run

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

Notes
-----

``SimpleWaveReader`` at the end of each program is a small helper for reading mono 16-bit WAVE files.

