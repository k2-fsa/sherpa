function(download_websocketpp)
  include(FetchContent)

  # The latest commit on the develop branch os as 2022-10-22
  set(websocketpp_URL  "https://github.com/zaphoyd/websocketpp/archive/b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
  set(websocketpp_HASH "SHA256=1385135ede8191a7fbef9ec8099e3c5a673d48df0c143958216cd1690567f583")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  if(EXISTS "/star-fj/fangjun/download/github/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
    set(websocketpp_URL  "file:///star-fj/fangjun/download/github/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
  elseif(EXISTS "/tmp/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
    set(websocketpp_URL  "file:///tmp/websocketpp-b9aeec6eaf3d5610503439b4fae3581d9aff08e8.zip")
  endif()

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
