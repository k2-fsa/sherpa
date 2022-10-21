function(download_websocketpp)
  if(CMAKE_VERSION VERSION_LESS 3.11)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
  endif()

  include(FetchContent)

  set(websocketpp_URL  "https://github.com/zaphoyd/websocketpp/archive/refs/tags/0.8.2.tar.gz")
  set(websocketpp_HASH "SHA256=6ce889d85ecdc2d8fa07408d6787e7352510750daa66b5ad44aacb47bea76755")

  FetchContent_Declare(websocketpp
    URL               ${websocketpp_URL}
    URL_HASH          ${websocketpp_HASH}
  )

  FetchContent_GetProperties(websocketpp)
  if(NOT websocketpp_POPULATED)
    message(STATUS "Downloading websocketpp ${websocketpp_URL}")
    FetchContent_Populate(websocketpp)
  endif()
  message(STATUS "websocketpp is downloaded to ${websocketpp_SOURCE_DIR}")
  # add_subdirectory(${websocketpp_SOURCE_DIR} ${websocketpp_BINARY_DIR} EXCLUDE_FROM_ALL)
  include_directories(${websocketpp_SOURCE_DIR})
endfunction()

download_websocketpp()
