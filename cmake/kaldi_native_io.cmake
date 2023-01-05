function(download_kaldi_native_io)
  include(FetchContent)

  set(kaldi_native_io_URL  "https://github.com/csukuangfj/kaldi_native_io/archive/refs/tags/v1.17.2.tar.gz")
  set(kaldi_native_io_HASH "SHA256=f916f2d3cd4c155b22cb64aa0d5e4f533b3ba2a40a77137c46506cfa7a00ec12")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  if(EXISTS "/star-fj/fangjun/download/github/kaldi_native_io-1.17.2.tar.gz")
    set(kaldi_native_io_URL  "file:///star-fj/fangjun/download/github/kaldi_native_io-1.17.2.tar.gz")
  elseif(EXISTS "/tmp/kaldi_native_io-1.17.2.tar.gz")
    set(kaldi_native_io_URL  "file:///tmp/kaldi_native_io-1.17.2.tar.gz")
  endif()

  set(KALDI_NATIVE_IO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_IO_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldi_native_io
    URL               ${kaldi_native_io_URL}
    URL_HASH          ${kaldi_native_io_HASH}
  )

  FetchContent_GetProperties(kaldi_native_io)
  if(NOT kaldi_native_io_POPULATED)
    message(STATUS "Downloading kaldi_native_io ${kaldi_native_io_URL}")
    FetchContent_Populate(kaldi_native_io)
  endif()
  message(STATUS "kaldi_native_io is downloaded to ${kaldi_native_io_SOURCE_DIR}")
  message(STATUS "kaldi_native_io's binary dir is ${kaldi_native_io_BINARY_DIR}")

  add_subdirectory(${kaldi_native_io_SOURCE_DIR} ${kaldi_native_io_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(kaldi_native_io_core
    PUBLIC
      ${kaldi_native_io_SOURCE_DIR}/
  )

  set_target_properties(kaldi_native_io_core PROPERTIES OUTPUT_NAME "sherpa_kaldi_native_io_core")

  install(TARGETS kaldi_native_io_core DESTINATION lib)
endfunction()

download_kaldi_native_io()
