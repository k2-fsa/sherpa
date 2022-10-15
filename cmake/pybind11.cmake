function(download_pybind11)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v2.10.0
  )

  FetchContent_GetProperties(pybind11)
  if(NOT pybind11_POPULATED)
    message(STATUS "Downloading pybind11")
    FetchContent_Populate(pybind11)
  endif()
  message(STATUS "pybind11 is downloaded to ${pybind11_SOURCE_DIR}")
  add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR} EXCLUDE_FROM_ALL)
endfunction()

download_pybind11()
