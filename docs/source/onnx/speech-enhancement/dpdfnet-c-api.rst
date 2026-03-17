DPDFNet C API
=============

This page describes how to use DPDFNet with the C API of `sherpa-onnx`_.

Please refer to :ref:`sherpa-onnx-c-api` for how to build `sherpa-onnx`_.

Offline speech enhancement
--------------------------

`sherpa-onnx`_ contains a ready-to-build example named
``speech-enhancement-dpdfnet-c-api.c``.

.. code-block:: bash

   cd /tmp
   git clone https://github.com/k2-fsa/sherpa-onnx
   cd sherpa-onnx

   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
   wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

   mkdir build
   cd build

   cmake \
     -DSHERPA_ONNX_ENABLE_C_API=ON \
     -DBUILD_SHARED_LIBS=ON \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_INSTALL_PREFIX=./install \
     ..

   make -j2 install
   cd ..

   gcc -o speech-enhancement-dpdfnet-c-api \
     ./c-api-examples/speech-enhancement-dpdfnet-c-api.c \
     -I ./build/install/include \
     -L ./build/install/lib \
     -l sherpa-onnx-c-api \
     -l onnxruntime

   export LD_LIBRARY_PATH=$PWD/build/install/lib:$LD_LIBRARY_PATH
   export DYLD_LIBRARY_PATH=$PWD/build/install/lib:$DYLD_LIBRARY_PATH

   ./speech-enhancement-dpdfnet-c-api

The example source is available at

  `<https://github.com/k2-fsa/sherpa-onnx/blob/master/c-api-examples/speech-enhancement-dpdfnet-c-api.c>`_

The core offline configuration is shown below:

.. code-block:: c

   #include <stdio.h>
   #include <string.h>

   #include "sherpa-onnx/c-api/c-api.h"

   int32_t main() {
     SherpaOnnxOfflineSpeechDenoiserConfig config;
     memset(&config, 0, sizeof(config));

     config.model.dpdfnet.model = "./dpdfnet2.onnx";
     config.model.num_threads = 1;
     config.model.debug = 0;
     config.model.provider = "cpu";

     const SherpaOnnxOfflineSpeechDenoiser *sd =
         SherpaOnnxCreateOfflineSpeechDenoiser(&config);

     const SherpaOnnxWave *wave = SherpaOnnxReadWave("./inp_16k.wav");
     const SherpaOnnxDenoisedAudio *denoised =
         SherpaOnnxOfflineSpeechDenoiserRun(
             sd, wave->samples, wave->num_samples, wave->sample_rate);

     SherpaOnnxWriteWave(
         denoised->samples, denoised->n, denoised->sample_rate,
         "./enhanced.wav");

     SherpaOnnxDestroyDenoisedAudio(denoised);
     SherpaOnnxFreeWave(wave);
     SherpaOnnxDestroyOfflineSpeechDenoiser(sd);
     return 0;
   }

Streaming speech enhancement
----------------------------

DPDFNet is also available through the streaming denoiser API:

  - ``SherpaOnnxOnlineSpeechDenoiserConfig``
  - ``SherpaOnnxCreateOnlineSpeechDenoiser()``
  - ``SherpaOnnxOnlineSpeechDenoiserRun()``
  - ``SherpaOnnxOnlineSpeechDenoiserFlush()``
  - ``SherpaOnnxOnlineSpeechDenoiserReset()``

The following example processes a wave file frame by frame and writes the
streaming output to ``enhanced-streaming.wav``:

.. code-block:: c

   #include <stdint.h>
   #include <stdio.h>
   #include <stdlib.h>
   #include <string.h>

   #include "sherpa-onnx/c-api/c-api.h"

   static int32_t AppendSamples(float **buffer, int32_t *size,
                                int32_t *capacity, const float *samples,
                                int32_t n) {
     if (*size + n > *capacity) {
       int32_t new_capacity = *capacity == 0 ? n : *capacity * 2;
       while (new_capacity < *size + n) {
         new_capacity *= 2;
       }

       float *new_buffer =
           (float *)realloc(*buffer, new_capacity * sizeof(float));
       if (new_buffer == NULL) {
         return 0;
       }

       *buffer = new_buffer;
       *capacity = new_capacity;
     }

     memcpy(*buffer + *size, samples, n * sizeof(float));
     *size += n;
     return 1;
   }

   int32_t main() {
     SherpaOnnxOnlineSpeechDenoiserConfig config;
     memset(&config, 0, sizeof(config));

     config.model.dpdfnet.model = "./dpdfnet2.onnx";
     config.model.num_threads = 1;
     config.model.debug = 0;
     config.model.provider = "cpu";

     const SherpaOnnxOnlineSpeechDenoiser *sd =
         SherpaOnnxCreateOnlineSpeechDenoiser(&config);
     if (sd == NULL) {
       fprintf(stderr, "Failed to create online speech denoiser\n");
       return -1;
     }

     const SherpaOnnxWave *wave = SherpaOnnxReadWave("./inp_16k.wav");
     if (wave == NULL) {
       SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
       fprintf(stderr, "Failed to read input wave\n");
       return -1;
     }

     const int32_t frame_shift =
         SherpaOnnxOnlineSpeechDenoiserGetFrameShiftInSamples(sd);

     float *enhanced = NULL;
     int32_t num_enhanced = 0;
     int32_t capacity = 0;

     for (int32_t start = 0; start < wave->num_samples; start += frame_shift) {
       int32_t n = wave->num_samples - start;
       if (n > frame_shift) {
         n = frame_shift;
       }

       const SherpaOnnxDenoisedAudio *chunk =
           SherpaOnnxOnlineSpeechDenoiserRun(
               sd, wave->samples + start, n, wave->sample_rate);
       if (chunk == NULL) {
         continue;
       }

       if (!AppendSamples(&enhanced, &num_enhanced, &capacity,
                          chunk->samples, chunk->n)) {
         fprintf(stderr, "Failed to grow output buffer\n");
         SherpaOnnxDestroyDenoisedAudio(chunk);
         SherpaOnnxFreeWave(wave);
         SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
         free(enhanced);
         return -1;
       }

       SherpaOnnxDestroyDenoisedAudio(chunk);
     }

     const SherpaOnnxDenoisedAudio *tail =
         SherpaOnnxOnlineSpeechDenoiserFlush(sd);
     if (tail != NULL) {
       AppendSamples(&enhanced, &num_enhanced, &capacity,
                     tail->samples, tail->n);
       SherpaOnnxDestroyDenoisedAudio(tail);
     }

     SherpaOnnxWriteWave(
         enhanced, num_enhanced,
         SherpaOnnxOnlineSpeechDenoiserGetSampleRate(sd),
         "./enhanced-streaming.wav");

     free(enhanced);
     SherpaOnnxFreeWave(wave);
     SherpaOnnxDestroyOnlineSpeechDenoiser(sd);
     return 0;
   }

.. note::

   ``SherpaOnnxOnlineSpeechDenoiserRun()`` can return ``NULL`` until enough
   input audio has been buffered. Call
   ``SherpaOnnxOnlineSpeechDenoiserFlush()`` at the end of the stream to
   retrieve the final tail samples and reset the denoiser state.
