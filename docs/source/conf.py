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
import os
import re
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "sherpa"
copyright = "2022, sherpa development team"
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
rst_epilog = """
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
.. _wenet: https://github.com/k2-fsa/sherpa
.. _GigaSpeech: https://github.com/SpeechColab/GigaSpeech
.. _Kaldi: https://github.com/kaldi-asr/kaldi
.. _kaldifeat: https://csukuangfj.github.io/kaldifeat/installation.html
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
"""
