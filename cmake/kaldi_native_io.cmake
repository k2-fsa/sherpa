function(download_kaldi_native_io)
  include(FetchContent)

  set(kaldi_native_io_URL  "https://github.com/csukuangfj/kaldi_native_io/archive/refs/tags/v1.17.2.tar.gz")
  set(kaldi_native_io_URL2 "https://huggingface.co/csukuangfj/sherpa-cmake-deps/resolve/main/kaldi_native_io-1.17.2.tar.gz")
  set(kaldi_native_io_HASH "SHA256=f916f2d3cd4c155b22cb64aa0d5e4f533b3ba2a40a77137c46506cfa7a00ec12")

  # If you don't have access to the Internet,
  # please pre-download kaldi_native_io
  set(possible_file_locations
    $ENV{HOME}/Downloads/kaldi_native_io-1.17.2.tar.gz
    ${PROJECT_SOURCE_DIR}/kaldi_native_io-1.17.2.tar.gz
    ${PROJECT_BINARY_DIR}/kaldi_native_io-1.17.2.tar.gz
    /tmp/kaldi_native_io-1.17.2.tar.gz
    /star-fj/fangjun/download/github/kaldi_native_io-1.17.2.tar.gz
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(kaldi_native_io_URL  "${f}")
      file(TO_CMAKE_PATH "${kaldi_native_io_URL}" kaldi_native_io_URL)
      set(kaldi_native_io_URL2)
      break()
    endif()
  endforeach()

  set(KALDI_NATIVE_IO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(KALDI_NATIVE_IO_BUILD_PYTHON OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(kaldi_native_io
    URL
      ${kaldi_native_io_URL}
      ${kaldi_native_io_URL2}
    URL_HASH          ${kaldi_native_io_HASH}
  )

  FetchContent_GetProperties(kaldi_native_io)
  if(NOT kaldi_native_io_POPULATED)
    message(STATUS "Downloading kaldi_native_io from ${kaldi_native_io_URL}")
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
