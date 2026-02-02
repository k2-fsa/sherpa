Build sherpa-onnx on Windows ({{ arch }})
=========================================

This section describes how to build **sherpa-onnx** on Windows for the
**{{ arch }}** architecture using CPU only.

You can choose your build configuration depending on:

- Runtime: MD (dynamic CRT), MT (static CRT)
- Library type: shared or static
- Build type: Debug, Release, MinSizeRel, RelWithDebInfo

Prerequisites
-------------

- Windows 10 or later
- **Visual Studio 2022** or newer
- CMake 3.20 or newer
- A C++ compiler provided by Visual Studio
- Optional: PortAudio and eSpeak NG (disabled in examples)

.. note::

   MinGW is not supported.

Build Configurations
-------------------

{% for lib_type in lib_types %}
{% for runtime in runtimes %}
{% for build_type in build_types %}
{{ lib_type | capitalize }} + {{ runtime }} + {{ build_type }}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build commands
^^^^^^^^^^^^^^

::

  mkdir build
  cd build
  cmake \
    -A {{ cmake_arch }} \
    -DSHERPA_ONNX_ENABLE_TTS=ON \
    -DSHERPA_ONNX_USE_STATIC_CRT={{ 'ON' if runtime=='MT' else 'OFF' }} \
    -DCMAKE_BUILD_TYPE={{ build_type }} \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DBUILD_SHARED_LIBS={{ 'ON' if lib_type=='shared' else 'OFF' }} \
    -DCMAKE_INSTALL_PREFIX=./install \
    -DBUILD_ESPEAK_NG_EXE=OFF \
    ..
  cmake --build . --config {{ build_type }} -- -m:2
  cmake --build . --config {{ build_type }} --target install -- -m:2

{% if build_type == "Release" %}
Check runtime dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  dumpbin /dependents build\bin\Release\sherpa-onnx.exe

Expected output
""""""""""""""""""""

::

{% if lib_type=='shared' and runtime=='MD' %}
  Dump of file build\bin\Release\sherpa-onnx.exe

  File Type: EXECUTABLE IMAGE

    Image has the following dependencies:

      onnxruntime.dll
      KERNEL32.dll
      MSVCP140.dll
      VCRUNTIME140.dll
      api-ms-win-crt-runtime-l1-1-0.dll
      api-ms-win-crt-heap-l1-1-0.dll
      api-ms-win-crt-stdio-l1-1-0.dll
      api-ms-win-crt-filesystem-l1-1-0.dll
      api-ms-win-crt-string-l1-1-0.dll
      api-ms-win-crt-convert-l1-1-0.dll
      api-ms-win-crt-math-l1-1-0.dll
      api-ms-win-crt-utility-l1-1-0.dll
      api-ms-win-crt-environment-l1-1-0.dll
      api-ms-win-crt-locale-l1-1-0.dll

{% elif lib_type=='static' and runtime=='MD' %}
  Dump of file build\bin\Release\sherpa-onnx.exe

  File Type: EXECUTABLE IMAGE

    Image has the following dependencies:

      KERNEL32.dll
      ADVAPI32.dll
      MSVCP140.dll
      MSVCP140_1.dll
      dbghelp.dll
      api-ms-win-core-path-l1-1-0.dll
      SETUPAPI.dll
      dxgi.dll
      VCRUNTIME140.dll
      api-ms-win-crt-stdio-l1-1-0.dll
      api-ms-win-crt-heap-l1-1-0.dll
      api-ms-win-crt-runtime-l1-1-0.dll
      api-ms-win-crt-string-l1-1-0.dll
      api-ms-win-crt-convert-l1-1-0.dll
      api-ms-win-crt-filesystem-l1-1-0.dll
      api-ms-win-crt-math-l1-1-0.dll
      api-ms-win-crt-utility-l1-1-0.dll
      api-ms-win-crt-time-l1-1-0.dll
      api-ms-win-crt-locale-l1-1-0.dll
      api-ms-win-crt-environment-l1-1-0.dll

{% elif lib_type=='shared' and runtime=='MT' %}
  Dump of file build\bin\Release\sherpa-onnx.exe

  File Type: EXECUTABLE IMAGE

    Image has the following dependencies:

      onnxruntime.dll
      KERNEL32.dll

{% elif lib_type=='static' and runtime=='MT' %}
  Dump of file build\bin\Release\sherpa-onnx.exe

  File Type: EXECUTABLE IMAGE

    Image has the following dependencies:

      KERNEL32.dll
      ADVAPI32.dll
      dbghelp.dll
      api-ms-win-core-path-l1-1-0.dll
      SETUPAPI.dll
      dxgi.dll

{% endif %}
{% endif %}   {# closes the build_type == Release if #}

{% endfor %}
{% endfor %}
{% endfor %}

.. note::

   - ``MT`` builds do **not** depend on the MSVC runtime DLLs.  
   - ``MD`` builds require the Microsoft Visual C++ Redistributable.  
   - ``shared`` builds depend on ``onnxruntime.dll`` at runtime.  
   - Users can enable or disable TTS by changing ``-DSHERPA_ONNX_ENABLE_TTS=ON`` or ``OFF``.

