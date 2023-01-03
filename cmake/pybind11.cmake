function(download_pybind11)
  include(FetchContent)

  set(pybind11_URL  "https://github.com/pybind/pybind11/archive/v2.9.2.tar.gz")
  set(pybind11_HASH "SHA256=6bd528c4dbe2276635dc787b6b1f2e5316cf6b49ee3e150264e455a0d68d19c1")

  FetchContent_Declare(pybind11
    URL               ${pybind11_URL}
    URL_HASH          ${pybind11_HASH}
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
