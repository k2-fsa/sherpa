.. _onnx-x-asr-swift-api:

Swift
=====

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Swift API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model. The Swift package manager (SPM) downloads a pre-built library automatically, so you don't need to build `sherpa-onnx`_ yourself.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Xcode 15 or the Swift toolchain 5.9 or above

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Package.swift``:

.. code-block:: swift

   // swift-tools-version:5.9
   import PackageDescription

   let package = Package(
     name: "XasrDemo",
     platforms: [.iOS(.v15), .macOS(.v10_15)],
     dependencies: [
       .package(url: "https://github.com/k2-fsa/sherpa-onnx", from: "2.0.1"),
     ],
     targets: [
       .executableTarget(
         name: "XasrDemo",
         dependencies: [
           .product(name: "sherpa-onnx", package: "sherpa-onnx"),
         ]
       )
     ]
   )

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Sources/XasrDemo/XasrOfflineDemo.swift``:

.. code-block:: swift

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in Swift.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   import AVFoundation
   import SherpaOnnx

   extension AudioBuffer {
     func array() -> [Float] {
       return Array(UnsafeBufferPointer(self))
     }
   }

   extension AVAudioPCMBuffer {
     func array() -> [Float] {
       return self.audioBufferList.pointee.mBuffers.array()
     }
   }

   func run() {
     let modelDir = "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"

     // Step 1: Create the recognizer from the X-ASR model files.
     let transducerConfig = sherpaOnnxOfflineTransducerModelConfig(
       encoder: "\(modelDir)/encoder-epoch-99-avg-1.int8.onnx",
       decoder: "\(modelDir)/decoder-epoch-99-avg-1.onnx",
       joiner: "\(modelDir)/joiner-epoch-99-avg-1.int8.onnx"
     )
     let modelConfig = sherpaOnnxOfflineModelConfig(
       tokens: "\(modelDir)/tokens.txt",
       transducer: transducerConfig,
       numThreads: 2
     )
     let featConfig = sherpaOnnxFeatureConfig(sampleRate: 16000, featureDim: 80)
     var config = sherpaOnnxOfflineRecognizerConfig(
       featConfig: featConfig,
       modelConfig: modelConfig
     )
     let recognizer = SherpaOnnxOfflineRecognizer(config: &config)

     // Step 2: Read the wave file you want to transcribe.
     let filename = "\(modelDir)/test_wavs/0.wav"
     let fileURL: NSURL = NSURL(fileURLWithPath: filename)
     let audioFile = try! AVAudioFile(forReading: fileURL as URL)
     let audioFormat = audioFile.processingFormat
     assert(audioFormat.channelCount == 1)
     assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)
     let audioFrameCount = UInt32(audioFile.length)
     let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)
     try! audioFile.read(into: audioFileBuffer!)
     let samples: [Float]! = audioFileBuffer?.array()

     // Step 3: Send the whole audio to the recognizer and decode.
     let start = Date()
     let result = recognizer.decode(samples: samples, sampleRate: Int(audioFormat.sampleRate))
     let elapsed = Date().timeIntervalSince(start)

     // Step 4: Print the result. X-ASR punct models also add punctuations.
     print(filename)
     print(result.text)

     let audioDuration = Double(samples.count) / audioFormat.sampleRate
     print(String(format: "audio duration: %.3f s", audioDuration))
     print(String(format: "elapsed seconds: %.3f s", elapsed))
     print(String(format: "RTF = %.3f/%.3f = %.3f", elapsed, audioDuration, elapsed / audioDuration))
   }

   @main
   struct App {
     static func main() {
       run()
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Package.swift
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
   └── Sources
       └── XasrDemo
           └── XasrOfflineDemo.swift

   5 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   403B 19 Nov 12:23 Package.swift
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 Sources

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   swift run

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.154 s
   RTF = 0.154/10.053 = 0.015

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

Save the following as ``Package.swift``:

.. code-block:: swift

   // swift-tools-version:5.9
   import PackageDescription

   let package = Package(
     name: "XasrDemo",
     platforms: [.iOS(.v15), .macOS(.v10_15)],
     dependencies: [
       .package(url: "https://github.com/k2-fsa/sherpa-onnx", from: "2.0.1"),
     ],
     targets: [
       .executableTarget(
         name: "XasrDemo",
         dependencies: [
           .product(name: "sherpa-onnx", package: "sherpa-onnx"),
         ]
       )
     ]
   )

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``Sources/XasrDemo/XasrStreamingDemo.swift``:

.. code-block:: swift

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in Swift.
   //
   // Before you run it, please download the model:
   //
   //   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   import AVFoundation
   import SherpaOnnx

   extension AudioBuffer {
     func array() -> [Float] {
       return Array(UnsafeBufferPointer(self))
     }
   }

   extension AVAudioPCMBuffer {
     func array() -> [Float] {
       return self.audioBufferList.pointee.mBuffers.array()
     }
   }

   func run() {
     let modelDir =
       "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"

     // Step 1: Create the recognizer from the X-ASR model files.
     let transducerConfig = sherpaOnnxOnlineTransducerModelConfig(
       encoder: "\(modelDir)/encoder.int8.onnx",
       decoder: "\(modelDir)/decoder.onnx",
       joiner: "\(modelDir)/joiner.int8.onnx"
     )
     let modelConfig = sherpaOnnxOnlineModelConfig(
       tokens: "\(modelDir)/tokens.txt",
       transducer: transducerConfig,
       numThreads: 2
     )
     let featConfig = sherpaOnnxFeatureConfig(sampleRate: 16000, featureDim: 80)
     var config = sherpaOnnxOnlineRecognizerConfig(
       featConfig: featConfig,
       modelConfig: modelConfig
     )
     let recognizer = SherpaOnnxRecognizer(config: &config)

     // Step 2: Read the wave file you want to transcribe.
     let filename = "\(modelDir)/test_wavs/0.wav"
     let fileURL: NSURL = NSURL(fileURLWithPath: filename)
     let audioFile = try! AVAudioFile(forReading: fileURL as URL)
     let audioFormat = audioFile.processingFormat
     assert(audioFormat.channelCount == 1)
     assert(audioFormat.commonFormat == AVAudioCommonFormat.pcmFormatFloat32)
     let audioFrameCount = UInt32(audioFile.length)
     let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)
     try! audioFile.read(into: audioFileBuffer!)
     let samples: [Float]! = audioFileBuffer?.array()

     // Step 3: Feed the audio in small chunks, just like a microphone would do
     // in a real-time application.
     let chunk = 3200  // 0.2 second for 16 kHz audio

     let start = Date()
     var last = ""
     var i = 0
     while i < samples.count {
       let end = min(i + chunk, samples.count)
       recognizer.acceptWaveform(samples: Array(samples[i..<end]))

       while recognizer.isReady() {
         recognizer.decode()
       }

       // Print the latest partial result when it changes.
       let text = recognizer.getResult().text
       if !text.isEmpty && text != last {
         last = text
         print(text)
       }
       i += chunk
     }

     // Send some tail padding so the last words are flushed out.
     let tailPadding = [Float](repeating: 0.0, count: Int(0.5 * audioFormat.sampleRate))
     recognizer.acceptWaveform(samples: tailPadding)
     recognizer.inputFinished()
     while recognizer.isReady() {
       recognizer.decode()
     }

     // Step 4: Print the final result.
     let elapsed = Date().timeIntervalSince(start)
     print("Final result: \(recognizer.getResult().text)")

     let audioDuration = Double(samples.count) / audioFormat.sampleRate
     print(String(format: "audio duration: %.3f s", audioDuration))
     print(String(format: "elapsed seconds: %.3f s", elapsed))
     print(String(format: "RTF = %.3f/%.3f = %.3f", elapsed, audioDuration, elapsed / audioDuration))
   }

   @main
   struct App {
     static func main() {
       run()
     }
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── Package.swift
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
   └── Sources
       └── XasrDemo
           └── XasrStreamingDemo.swift

   5 directories, 13 files

   $ ls -lh
   total 8
   -rw-r--r--@  1 fangjun  wheel   403B 19 Nov 12:23 Package.swift
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05
   drwxr-xr-x@  3 fangjun  wheel    96B 19 Nov 12:23 Sources

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

From the project directory, run:

.. code-block:: bash

   swift run

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
   elapsed seconds: 0.322 s
   RTF = 0.322/10.053 = 0.032

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

``from: "2.0.1"`` pins the version of `sherpa-onnx`_; please use a version at least as new as the model files you downloaded.

