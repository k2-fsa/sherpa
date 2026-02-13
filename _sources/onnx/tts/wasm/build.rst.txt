Build
=====

After installing `emscripten`_, we can build text-to-speech from
`sherpa-onnx`_ for `WebAssembly`_ now.

Please use the following command to build it:

.. code-block:: bash

  git clone https://github.com/k2-fsa/sherpa-onnx
  cd sherpa-onnx

  cd wasm/tts/assets

  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
  tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
  rm vits-piper-en_US-libritts_r-medium.tar.bz2
  mv vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx ./model.onnx
  mv vits-piper-en_US-libritts_r-medium/tokens.txt ./
  mv vits-piper-en_US-libritts_r-medium/espeak-ng-data ./
  rm -rf vits-piper-en_US-libritts_r-medium

  cd ../../..

  ./build-wasm-simd-tts.sh

.. hint::

   You can visit `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>`_
   to download a different model.

After building, you should see the following output:

.. code-block:: bash

  Install the project...
  -- Install configuration: "Release"
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libkaldi-native-fbank-core.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libkaldi-decoder-core.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libsherpa-onnx-kaldifst-core.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libsherpa-onnx-fst.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libonnxruntime.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libespeak-ng.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libucd.a
  -- Up-to-date: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libucd.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libpiper_phonemize.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/./sherpa-onnx.pc
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/pkgconfig/espeak-ng.pc
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/share/vim/vimfiles/ftdetect
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/share/vim/vimfiles/ftdetect/espeakfiletype.vim
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/share/vim/vimfiles/syntax
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/share/vim/vimfiles/syntax/espeakrules.vim
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/share/vim/vimfiles/syntax/espeaklist.vim
  -- Up-to-date: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libucd.a
  -- Up-to-date: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libespeak-ng.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libsherpa-onnx-core.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/lib/libsherpa-onnx-c-api.a
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/include/sherpa-onnx/c-api/c-api.h
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/sherpa-onnx-wasm-main.js
  -- Up-to-date: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/sherpa-onnx-wasm-main.js
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/index.html
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/sherpa-onnx.js
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/app.js
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/sherpa-onnx-wasm-main.wasm
  -- Installing: /Users/fangjun/open-source/sherpa-onnx/build-wasm-simd-tts/install/bin/wasm/tts/sherpa-onnx-wasm-main.data
  + ls -lh install/bin/wasm/tts
  total 211248
  -rw-r--r--  1 fangjun  staff   5.3K Feb 22 09:18 app.js
  -rw-r--r--  1 fangjun  staff   1.3K Feb 22 09:18 index.html
  -rw-r--r--  1 fangjun  staff    92M Feb 22 10:35 sherpa-onnx-wasm-main.data
  -rw-r--r--  1 fangjun  staff   117K Feb 22 10:39 sherpa-onnx-wasm-main.js
  -rw-r--r--  1 fangjun  staff    11M Feb 22 10:39 sherpa-onnx-wasm-main.wasm
  -rw-r--r--  1 fangjun  staff   4.5K Feb 22 09:18 sherpa-onnx.js

Now you can use the following command to run it:

.. code-block:: bash

  cd build-wasm-simd-tts/install/bin/wasm/tts
  python3 -m http.server 6008

Start your browser and visit `<http://localhost:6008/>`_; you should see the following
page:

.. figure:: ./pic/wasm-sherpa-onnx-tts-1.png
   :alt: start page of wasm
   :width: 800

Now you can enter some text and click ``Generate``

A screenshot is given below:

.. figure:: ./pic/wasm-sherpa-onnx-tts-2.png
   :alt: tts result
   :width: 800

Congratulations! You have successfully run text-to-speech with `WebAssembly`_
in your browser.
