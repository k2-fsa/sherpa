function(download_brpc)
  include(FetchContent)
  FetchContent_Declare(brpc
    GIT_REPOSITORY https://github.com/apache/incubator-brpc
    GIT_TAG 1.1.0
  )
  FetchContent_GetProperties(brpc)
  if(NOT brpc_POPULATED)
    FetchContent_Populate(brpc)
  endif()
  add_subdirectory(${brpc_SOURCE_DIR} ${brpc_BINARY_DIR})
  include_directories(${brpc_SOURCE_DIR}/src ${brpc_BINARY_DIR})
endfunction()

download_brpc()
