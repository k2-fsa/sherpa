function(download_grpc)
  message(STATUS "Using gRPC via add_subdirectory")
  include(FetchContent)
  #SET(CMAKE_CXX_FLAGS  "-DBUILD_SHARED_LIBS=ON")

  set(ABSL_ENABLE_INSTALL ON)
  FetchContent_Declare(gRPC
    GIT_REPOSITORY https://github.com/grpc/grpc
    GIT_TAG        v1.57.0
  )
  set(FETCHCONTENT_QUIET OFF)
  FetchContent_MakeAvailable(gRPC)

  message(STATUS "grpc is downloaded to ${grpc_SOURCE_DIR}")
endfunction()

download_grpc()
