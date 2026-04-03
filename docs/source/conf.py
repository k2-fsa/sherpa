# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import re
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

year = datetime.date.today().year
project = "sherpa"
copyright = f"2022-{year}, sherpa development team"
author = "sherpa development team"


def get_version():
    cmake_file = "../../CMakeLists.txt"
    with open(cmake_file) as f:
        content = f.read()

    version = re.search(r"set\(SHERPA_VERSION (.*)\)", content).group(1)
    return version.strip('"')


# The full version, including alpha/beta/rc tags
version = get_version()
release = version
sherpa_onnx_version = os.environ.get("SHERPA_ONNX_VERSION", "1.12.32")
sherpa_onnx_version_cuda = f"{sherpa_onnx_version}+cuda"
sherpa_onnx_version_cuda12_cudnn9 = f"{sherpa_onnx_version}+cuda12.cudnn9"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "_ext.rst_roles",
    "recommonmark",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx_tabs.tabs",
    "sphinxcontrib.youtube",
    "sphinx_copybutton",  # https://sphinx-copybutton.readthedocs.io/en/latest/
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["python/installation/pic/*.md", "_ext/README.md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_show_sourcelink = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["css/user.define.css"]

pygments_style = "sphinx"
numfig = True

html_context = {
    "display_github": True,
    "github_user": "k2-fsa",
    "github_repo": "sherpa",
    "github_version": "master",
    "conf_py_path": "/docs/source/",
}

# refer to
# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}
rst_epilog = f"""
.. _BPE: https://arxiv.org/pdf/1508.07909v5.pdf
.. _Conformer: https://arxiv.org/abs/2005.08100
.. _ConvEmformer: https://arxiv.org/pdf/2110.05241.pdf
.. _Emformer: https://arxiv.org/pdf/2010.10759.pdf
.. _LibriSpeech: https://www.openslr.org/12
.. _CSJ: https://clrd.ninjal.ac.jp/csj/en/index.html
.. _aishell: https://www.openslr.org/33
.. _sherpa: https://github.com/k2-fsa/sherpa
.. _transducer: https://arxiv.org/pdf/1211.3711.pdf
.. _CTC: https://www.cs.toronto.edu/~graves/icml_2006.pdf
.. _asyncio: https://docs.python.org/3/library/asyncio.html
.. _k2: https://github.com/k2-fsa/k2
.. _icefall: https://github.com/k2-fsa/icefall
.. _PyTorch: https://pytorch.org/
.. _Huggingface: https://huggingface.co
.. _WenetSpeech: https://github.com/wenet-e2e/WenetSpeech
.. _WeNet: https://github.com/wenet-e2e/wenet
.. _GigaSpeech: https://github.com/SpeechColab/GigaSpeech
.. _Kaldi: https://github.com/kaldi-asr/kaldi
.. _kaldifeat: https://csukuangfj.github.io/kaldifeat/installation/index.html
.. _ncnn: https://github.com/tencent/ncnn
.. _sherpa-ncnn: https://github.com/k2-fsa/sherpa-ncnn
.. _onnx: https://github.com/onnx/onnx
.. _onnxruntime: https://github.com/microsoft/onnxruntime
.. _sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
.. _torchaudio: https://github.com/pytorch/audio
.. _Docker: https://www.docker.com
.. _Triton: https://github.com/triton-inference-server
.. _Triton-server: https://github.com/triton-inference-server/server
.. _Triton-client: https://github.com/triton-inference-server/client
.. _WebSocket: https://en.wikipedia.org/wiki/WebSocket
.. _websocketpp: https://github.com/zaphoyd/websocketpp
.. _asio: https://github.com/chriskohlhoff/asio
.. _boost: https://github.com/boostorg/boost
.. _NeMo: https://github.com/NVIDIA/NeMo
.. _CommonVoice: https://commonvoice.mozilla.org
.. _Zipformer: https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming
.. _VisionFive2: https://www.starfivetech.com/en/site/boards
.. _k2-fsa/sherpa: http://github.com/k2-fsa/sherpa
.. _k2-fsa/sherpa-onnx: http://github.com/k2-fsa/sherpa-onnx
.. _k2-fsa/sherpa-ncnn: http://github.com/k2-fsa/sherpa-ncnn
.. _srs: https://github.com/ossrs/srs
.. _RTMP: https://en.wikipedia.org/wiki/Real-Time_Messaging_Protocol
.. _Whisper: https://github.com/openai/whisper/
.. _Go: https://en.wikipedia.org/wiki/Go_(programming_language)
.. _sherpa-onnx-go: https://github.com/k2-fsa/sherpa-onnx-go
.. _yesno: https://www.openslr.org/1/
.. _vits: https://github.com/jaywalnut310/vits
.. _ljspeech: https://keithito.com/LJ-Speech-Dataset/
.. _LJ Speech: https://keithito.com/LJ-Speech-Dataset/
.. _VCTK: https://datashare.ed.ac.uk/handle/10283/2950
.. _piper: https://github.com/rhasspy/piper
.. _aishell3: https://www.openslr.org/93/
.. _lessac_blizzard2013: https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/
.. _OpenFst: https://www.openfst.org/
.. _MMS: https://huggingface.co/spaces/mms-meta/MMS
.. _WebAssembly: https://en.wikipedia.org/wiki/WebAssembly
.. _emscripten: https://emscripten.org/index.html
.. _audioset: https://research.google.com/audioset/
.. _silero-vad: https://github.com/snakers4/silero-vad
.. _silero_vad: https://github.com/snakers4/silero-vad
.. _ten-vad: https://github.com/TEN-framework/ten-vad
.. _Flutter: https://flutter.dev/
.. _Dart: https://dart.dev/
.. _Node: https://nodejs.org/en
.. _SenseVoice: https://github.com/FunAudioLLM/SenseVoice
.. _LibriTTS-R: https://www.openslr.org/141/
.. _ReazonSpeech: https://github.com/reazon-research/ReazonSpeech
.. _Lazarus: https://www.lazarus-ide.org/
.. _Moonshine: https://github.com/usefulsensors/moonshine
.. _moonshine: https://github.com/usefulsensors/moonshine
.. _Omnilingual ASR: https://github.com/facebookresearch/omnilingual-asr
.. _FireRedAsr: https://github.com/FireRedTeam/FireRedASR
.. _Dolphin: https://github.com/DataoceanAI/Dolphin
.. _k2-fsa: https://github.com/k2-fsa
.. _Fun-ASR-Nano-2512: https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512
.. |sherpa_onnx_version| replace:: {sherpa_onnx_version}
.. |sherpa_onnx_version_cuda| replace:: {sherpa_onnx_version_cuda}
.. |sherpa_onnx_version_cuda12_cudnn9| replace:: {sherpa_onnx_version_cuda12_cudnn9}
.. |sherpa_onnx_release_version| replace:: ``v{sherpa_onnx_version}``
.. |sherpa_onnx_release_tag_url| replace:: https://github.com/k2-fsa/sherpa-onnx/releases/tag/v{sherpa_onnx_version}
.. |sherpa_onnx_crate_dependency| replace:: sherpa-onnx = "{sherpa_onnx_version}"
.. |sherpa_onnx_crate_shared_dependency| replace:: sherpa-onnx = {{ version = "{sherpa_onnx_version}", default-features = false, features = ["shared"] }}
.. |sherpa_onnx_version_output| replace:: Version : {sherpa_onnx_version}
.. |sherpa_onnx_linux_static_lib_dir| replace:: /path/to/sherpa-onnx-v{sherpa_onnx_version}-linux-x64-static-lib/lib
.. |sherpa_onnx_android_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-android.tar.bz2``
.. |sherpa_onnx_android_rknn_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-android-rknn.tar.bz2``
.. |sherpa_onnx_android_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-android.tar.bz2
.. |sherpa_onnx_android_rknn_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-android-rknn.tar.bz2
.. |sherpa_onnx_android_archive_url| replace:: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-android.tar.bz2
.. |sherpa_onnx_cuda_linux_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2``
.. |sherpa_onnx_cuda_linux_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2
.. |sherpa_onnx_cuda_linux_archive_url| replace:: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2
.. |sherpa_onnx_cuda_windows_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2``
.. |sherpa_onnx_cuda_windows_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2
.. |sherpa_onnx_cuda_windows_archive_url| replace:: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-cuda-12.x-cudnn-9.x-win-x64-cuda.tar.bz2
.. |sherpa_onnx_rknn_static_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-static.tar.bz2``
.. |sherpa_onnx_rknn_shared_archive_name| replace:: ``sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-shared.tar.bz2``
.. |sherpa_onnx_rknn_static_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-static.tar.bz2
.. |sherpa_onnx_rknn_shared_archive_file| replace:: sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-shared.tar.bz2
.. |sherpa_onnx_rknn_shared_dir| replace:: sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-shared
.. |sherpa_onnx_rknn_shared_archive_url| replace:: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-shared.tar.bz2
.. |sherpa_onnx_rknn_wheel_file| replace:: sherpa_onnx-{sherpa_onnx_version}-cp310-cp310-manylinux_2_27_aarch64.whl
.. |sherpa_onnx_rknn_installed| replace:: Successfully installed sherpa-onnx-{sherpa_onnx_version}
.. |sherpa_onnx_python_cpu_bin_wheel_url| replace:: https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/{sherpa_onnx_version}/sherpa_onnx_bin-{sherpa_onnx_version}-py3-none-manylinux2014_x86_64.whl
.. |sherpa_onnx_python_cpu_core_wheel_url| replace:: https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/{sherpa_onnx_version}/sherpa_onnx_core-{sherpa_onnx_version}-py3-none-manylinux2014_x86_64.whl
.. |sherpa_onnx_python_cpu_wheel_url| replace:: https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cpu/{sherpa_onnx_version}/sherpa_onnx-{sherpa_onnx_version}-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
.. |sherpa_onnx_python_cuda_spec| replace:: sherpa-onnx=="{sherpa_onnx_version_cuda}"
.. |sherpa_onnx_python_cuda_collecting| replace:: Collecting sherpa-onnx=={sherpa_onnx_version_cuda}
.. |sherpa_onnx_python_cuda_wheel_url| replace:: https://huggingface.co/csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/{sherpa_onnx_version}/sherpa_onnx-{sherpa_onnx_version}%2Bcuda-cp312-cp312-linux_x86_64.whl
.. |sherpa_onnx_python_cuda12_spec| replace:: sherpa-onnx=={sherpa_onnx_version_cuda12_cudnn9}
.. |sherpa_onnx_python_cuda12_collecting| replace:: Collecting sherpa-onnx=={sherpa_onnx_version_cuda12_cudnn9}
.. |sherpa_onnx_python_cuda12_wheel_url| replace:: https://huggingface.co/csukuangfj/sherpa-onnx-wheels/resolve/main/cuda/{sherpa_onnx_version}/sherpa_onnx-{sherpa_onnx_version}%2Bcuda12.cudnn9-cp312-cp312-linux_x86_64.whl
.. |sherpa_onnx_python_cuda12_installed| replace:: Successfully installed sherpa-onnx-{sherpa_onnx_version_cuda12_cudnn9}
.. _sherpa_onnx_release_tag: https://github.com/k2-fsa/sherpa-onnx/releases/tag/v{sherpa_onnx_version}
.. _sherpa_onnx_rknn_static_archive: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-static.tar.bz2
.. _sherpa_onnx_rknn_shared_archive: https://github.com/k2-fsa/sherpa-onnx/releases/download/v{sherpa_onnx_version}/sherpa-onnx-v{sherpa_onnx_version}-rknn-linux-aarch64-shared.tar.bz2
.. _Cohere Transcribe: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
"""


# https://docs.readthedocs.com/platform/latest/guides/pdf-non-ascii-languages.html
latex_engine = "xelatex"
latex_use_xindy = False
latex_elements = {
    "preamble": "\\usepackage[UTF8]{ctex}\n",
}


def setup(app):
    app.add_css_file("custom.css")
