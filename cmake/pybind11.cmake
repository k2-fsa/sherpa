function(download_pybind11)
  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/5bc0943ed96836f46489f53961f6c438d2935357.zip")
  set(pybind11_URL2 "https://huggingface.co/csukuangfj/k2-cmake-deps/resolve/main/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip")
  set(pybind11_HASH "SHA256=ff65a1a8c9e6ceec11e7ed9d296f2e22a63e9ff0c4264b3af29c72b4f18f25a0")

  # If you don't have access to the Internet,
  # please pre-download pybind11
  set(possible_file_locations
    $ENV{HOME}/Downloads/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip
    ${PROJECT_SOURCE_DIR}/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip
    ${PROJECT_BINARY_DIR}/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip
    /tmp/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip
    /star-fj/fangjun/download/github/pybind11-5bc0943ed96836f46489f53961f6c438d2935357.zip
  )

  foreach(f IN LISTS possible_file_locations)
    if(EXISTS ${f})
      set(pybind11_URL  "${f}")
      file(TO_CMAKE_PATH "${pybind11_URL}" pybind11_URL)
      set(pybind11_URL2)
      break()
    endif()
  endforeach()

  FetchContent_Declare(pybind11
    URL
      ${pybind11_URL}
      ${pybind11_URL2}
    URL_HASH          ${pybind11_HASH}
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11 from ${pybind11_URL}")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
