function(download_json)
  include(FetchContent)

  set(json_URL  "https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz")
  set(json_HASH "SHA256=d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273")

  FetchContent_Declare(json
    URL               ${json_URL}
    URL_HASH          ${json_HASH}
  )

  FetchContent_GetProperties(json)
  if(NOT json_POPULATED)
    message(STATUS "Downloading json ${json_URL}")
    FetchContent_Populate(json)
  endif()
  message(STATUS "json is downloaded to ${json_SOURCE_DIR}")
  include_directories(${json_SOURCE_DIR}/include)
  # Use #include "nlohmann/json.hpp"
endfunction()

download_json()
