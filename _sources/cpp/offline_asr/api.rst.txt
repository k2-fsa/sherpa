C++ APIs
========

We provide C++ APIs for non-streaming ASR
in
`a single header file <https://github.com/k2-fsa/sherpa/blob/master/sherpa/cpp_api/offline_recognizer.h>`_.

`<https://github.com/k2-fsa/sherpa-torch-cpp-makefile-example>`_
gives an example of how to use the APIs to decode wave files.

.. hint::

  You can find more examples in
  `<https://github.com/k2-fsa/sherpa/tree/master/sherpa/cpp_api>`_

The content of the `Makefile <https://github.com/k2-fsa/sherpa-torch-cpp-makefile-example/blob/master/Makefile>`_
from the above repository is given below:

.. code-block:: makefile

  sherpa_install_dir := $(shell python3 -c 'import os; import sherpa; print(os.path.dirname(sherpa.__file__))')
  sherpa_cxx_flags := $(shell python3 -c 'import os; import sherpa; print(sherpa.cxx_flags)')

  $(info sherpa_install_dir: $(sherpa_install_dir))
  $(info sherpa_cxx_flags: $(sherpa_cxx_flags))

  CXXFLAGS := -I$(sherpa_install_dir)/include
  CXXFLAGS += -Wl,-rpath,$(sherpa_install_dir)/lib
  CXXFLAGS += $(sherpa_cxx_flags)
  CXXFLAGS += -std=c++14

  LDFLAGS := -L $(sherpa_install_dir)/lib -lsherpa_offline_recognizer

  $(info CXXFLAGS: $(CXXFLAGS))
  $(info LDFLAGS: $(LDFLAGS))

  test_decode_files: test_decode_files.cc
    $(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

  .PHONY: clean
  clean:
    $(RM) test_decode_files

Basically, you only need to specify the following stuff to use the non-streaming
ASR APIs:

- The path to find the header file. You provide:

  .. code-block:: makefile

    CXXFLAGS := -I$(sherpa_install_dir)/include

- The library to link to. You provide:

  .. code-block:: makefile

    LDFLAGS := -L $(sherpa_install_dir)/lib -lsherpa_offline_recognizer

- The CXX flags used to compile ``sherpa``. You provide:

  .. code-block:: makefile

    CXXFLAGS += $(sherpa_cxx_flags)

  .. hint::

    This one is important. For instance, if ``sherpa`` was compiled with
    ``-D_GLIBCXX_USE_CXX11_ABI=0`` on Linux, you will get link errors like
    the below one if you don't use this option in your project.

    .. code-block::

      test_decode_files.cc:(.text+0x149): undefined reference to
      `sherpa::OfflineRecognizer::OfflineRecognizer(std::__cxx11::basic_string<char,
      std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char,
      std::char_traits<char>, std::allocator<char> > const&, sherpa::DecodingOptions const&, bool, float)'
