.. _onnx-x-asr-go-api:

Go
==

This page shows how to use `X-ASR <https://github.com/Gilgamesh-J/X-ASR>`_ models with the Go API of `sherpa-onnx`_. Two demos are given: one with a non-streaming model and one with a streaming model.

.. contents:: On this page
   :local:
   :depth: 2

Requirements
------------

- Go 1.21 or above
- ``CGO_ENABLED=1`` (it is required by the native library)

Non-streaming X-ASR
-------------------

Step 1. Create the project directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   mkdir x-asr-demo
   cd x-asr-demo

Step 2. Create the build file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Save the following as ``go.mod``:

.. code-block:: go

   module x-asr-demo

   go 1.21

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   rm sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``main.go``:

.. code-block:: go

   // This file shows how to use a non-streaming X-ASR model with sherpa-onnx
   // in Go.
   //
   // Before you run it, please download the model:
   //
   //	wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   //	tar xf sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03.tar.bz2
   package main

   import (
   	"fmt"
   	"log"
   	"time"

   	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
   )

   func main() {
   	modelDir := "./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03"

   	// Step 1: Create the recognizer from the X-ASR model files.
   	config := sherpa.OfflineRecognizerConfig{}
   	config.ModelConfig.Transducer.Encoder = modelDir + "/encoder-epoch-99-avg-1.int8.onnx"
   	config.ModelConfig.Transducer.Decoder = modelDir + "/decoder-epoch-99-avg-1.onnx"
   	config.ModelConfig.Transducer.Joiner = modelDir + "/joiner-epoch-99-avg-1.int8.onnx"
   	config.ModelConfig.Tokens = modelDir + "/tokens.txt"
   	config.ModelConfig.NumThreads = 2
   	config.ModelConfig.Debug = 0
   	config.DecodingMethod = "greedy_search"

   	recognizer := sherpa.NewOfflineRecognizer(&config)
   	defer sherpa.DeleteOfflineRecognizer(recognizer)

   	// Step 2: Read the wave file you want to transcribe.
   	filename := modelDir + "/test_wavs/0.wav"
   	wave := sherpa.ReadWave(filename)
   	if wave == nil {
   		log.Fatalf("Failed to read %v", filename)
   	}

   	// Step 3: Create a stream, send the whole audio to it, and decode.
   	start := time.Now()
   	stream := sherpa.NewOfflineStream(recognizer)
   	defer sherpa.DeleteOfflineStream(stream)
   	stream.AcceptWaveform(wave.SampleRate, wave.Samples)
   	recognizer.Decode(stream)

   	// Step 4: Print the result. X-ASR punct models also add punctuations.
   	fmt.Println(filename)
   	elapsed := time.Since(start).Seconds()
   	fmt.Println(stream.GetResult().Text)

   	audioDuration := float64(len(wave.Samples)) / float64(wave.SampleRate)
   	fmt.Printf("audio duration: %.3f s\n", audioDuration)
   	fmt.Printf("elapsed seconds: %.3f s\n", elapsed)
   	fmt.Printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audioDuration, elapsed/audioDuration)
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── go.mod
   ├── main.go
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

   3 directories, 13 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel    26B 19 Nov 12:23 go.mod
   -rw-r--r--@  1 fangjun  wheel   2.1K 19 Nov 12:23 main.go
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:31 sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   go mod tidy
   go run .

Output
^^^^^^

.. code-block:: text

   ./sherpa-onnx-x-asr-zipformer-transducer-zh-en-punct-int8-2026-06-03/test_wavs/0.wav
   昨天是 Monday, today is 礼拜二， the day after tomorrow 是星期三。
   audio duration: 10.053 s
   elapsed seconds: 0.108 s
   RTF = 0.108/10.053 = 0.011

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

Save the following as ``go.mod``:

.. code-block:: go

   module x-asr-demo

   go 1.21

Step 3. Download the model
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   rm sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2

Step 4. Save the code
^^^^^^^^^^^^^^^^^^^^^

Save the following as ``main.go``:

.. code-block:: go

   // This file shows how to use a streaming X-ASR model with sherpa-onnx
   // in Go.
   //
   // Before you run it, please download the model:
   //
   //	wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   //	tar xf sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05.tar.bz2
   package main

   import (
   	"fmt"
   	"log"
   	"time"

   	sherpa "github.com/k2-fsa/sherpa-onnx-go/v2/sherpa_onnx"
   )

   func main() {
   	modelDir := "./sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05"

   	// Step 1: Create the recognizer from the X-ASR model files.
   	config := sherpa.OnlineRecognizerConfig{}
   	config.ModelConfig.Transducer.Encoder = modelDir + "/encoder.int8.onnx"
   	config.ModelConfig.Transducer.Decoder = modelDir + "/decoder.onnx"
   	config.ModelConfig.Transducer.Joiner = modelDir + "/joiner.int8.onnx"
   	config.ModelConfig.Tokens = modelDir + "/tokens.txt"
   	config.ModelConfig.NumThreads = 2
   	config.ModelConfig.Debug = 0
   	config.DecodingMethod = "greedy_search"

   	recognizer := sherpa.NewOnlineRecognizer(&config)
   	defer sherpa.DeleteOnlineRecognizer(recognizer)

   	// Step 2: Read the wave file you want to transcribe.
   	filename := modelDir + "/test_wavs/0.wav"
   	wave := sherpa.ReadWave(filename)
   	if wave == nil {
   		log.Fatalf("Failed to read %v", filename)
   	}

   	// Step 3: Create a stream and feed the audio in small chunks, just like
   	// a microphone would do in a real-time application.
   	start := time.Now()
   	stream := sherpa.NewOnlineStream(recognizer)
   	defer sherpa.DeleteOnlineStream(stream)

   	// 0.2 second for 16 kHz audio
   	chunk := 3200

   	last := ""
   	for i := 0; i < len(wave.Samples); i += chunk {
   		end := i + chunk
   		if end > len(wave.Samples) {
   			end = len(wave.Samples)
   		}
   		stream.AcceptWaveform(wave.SampleRate, wave.Samples[i:end])

   		for recognizer.IsReady(stream) {
   			recognizer.Decode(stream)
   		}

   		// Print the latest partial result when it changes.
   		result := recognizer.GetResult(stream)
   		if result.Text != "" && result.Text != last {
   			last = result.Text
   			fmt.Println(result.Text)
   		}
   	}

   	// Send some tail padding so the last words are flushed out.
   	tail := make([]float32, int(0.5*float32(wave.SampleRate)))
   	stream.AcceptWaveform(wave.SampleRate, tail)
   	for recognizer.IsReady(stream) {
   		recognizer.Decode(stream)
   	}

   	// Step 4: Print the final result.
   	elapsed := time.Since(start).Seconds()
   	fmt.Println("Final result:", recognizer.GetResult(stream).Text)

   	audioDuration := float64(len(wave.Samples)) / float64(wave.SampleRate)
   	fmt.Printf("audio duration: %.3f s\n", audioDuration)
   	fmt.Printf("elapsed seconds: %.3f s\n", elapsed)
   	fmt.Printf("RTF = %.3f/%.3f = %.3f\n", elapsed, audioDuration, elapsed/audioDuration)
   }

Step 5. Check the project layout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your project directory should now look like this:

.. code-block:: text

   $ tree .
   .
   ├── go.mod
   ├── main.go
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

   3 directories, 13 files

   $ ls -lh
   total 16
   -rw-r--r--@  1 fangjun  wheel    26B 19 Nov 12:23 go.mod
   -rw-r--r--@  1 fangjun  wheel   2.7K 19 Nov 12:23 main.go
   drwxr-xr-x@ 10 fangjun  wheel   320B 29 Jun 16:30 sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-punct-int8-2026-06-05

Step 6. Build and run
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   go mod tidy
   go run .

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
   elapsed seconds: 0.298 s
   RTF = 0.298/10.053 = 0.030

.. note::

   The timing above was measured on a MacBook Pro (Apple M4) using 2 threads. Please increase the number of threads for a smaller RTF on your machine.

Notes
-----

If you see the error ``build constraints exclude all Go files``, please run ``go env -w CGO_ENABLED=1`` and try again.

