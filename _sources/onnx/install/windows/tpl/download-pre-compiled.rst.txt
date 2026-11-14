Pre-compiled executables and libraries for Windows {{arch}}
======================================================================

.. hint::

   If you don't want to build `sherpa-onnx`_ from source for Windows {{arch}}, please
   continue reading.


This section describes how to install prebuilt **sherpa-onnx** binaries on
**Windows {{arch}}**.

Please visit

  `<https://github.com/k2-fsa/sherpa-onnx/releases/tag/v{{sherpa_onnx_version}}>`_

to download pre-compiled executables and libraries for Windows ``{{arch}}``.


Shared Libraries
----------------

.. list-table::
   :header-rows: 1

   * - Runtime
     - Build Type
     - TTS
     - Download
{% for runtime in ["MT", "MD"] %}
{% for build in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"] %}
   * - {{ runtime }}
     - {{ build }}
     - disabled
     - `sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-shared-{{ runtime }}-{{ build }}-no-tts.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v{{sherpa_onnx_version}}/sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-shared-{{ runtime }}-{{ build }}-no-tts.tar.bz2>`_
   * - {{ runtime }}
     - {{ build }}
     - enabled
     - `sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-shared-{{ runtime }}-{{ build }}.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v{{sherpa_onnx_version}}/sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-shared-{{ runtime }}-{{ build }}.tar.bz2>`_
{% endfor %}
{% endfor %}

Static Libraries
----------------

.. list-table::
   :header-rows: 1

   * - Runtime
     - Build Type
     - TTS
     - Download
{% for runtime in ["MT", "MD"] %}
{% for build in ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"] %}
   * - {{ runtime }}
     - {{ build }}
     - disabled
     - `sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-static-{{ runtime }}-{{ build }}-no-tts.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v{{sherpa_onnx_version}}/sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-static-{{ runtime }}-{{ build }}-no-tts.tar.bz2>`_
   * - {{ runtime }}
     - {{ build }}
     - enabled
     - `sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-static-{{ runtime }}-{{ build }}.tar.bz2 <https://github.com/k2-fsa/sherpa-onnx/releases/download/v{{sherpa_onnx_version}}/sherpa-onnx-v{{sherpa_onnx_version}}-win-{{arch}}-static-{{ runtime }}-{{ build }}.tar.bz2>`_
{% endfor %}
{% endfor %}

.. note::

   All Windows {{arch}} binaries are built using **Visual Studio 2022**.

