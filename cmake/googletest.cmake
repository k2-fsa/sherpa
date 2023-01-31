function(download_googltest)
  include(FetchContent)

  set(googletest_URL  "https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz")
  set(googletest_HASH "SHA256=ad7fdba11ea011c1d925b3289cf4af2c66a352e18d4c7264392fead75e919363")

  # If you don't have access to the Internet, please download it to your
  # local drive and modify the following line according to your needs.
  if(EXISTS "/star-fj/fangjun/download/github/googletest-1.13.0.tar.gz")
    set(googletest_URL  "file:///star-fj/fangjun/download/github/googletest-1.13.0.tar.gz")
  elseif(EXISTS "/tmp/googletest-1.13.0.tar.gz")
    set(googletest_URL  "file:///tmp/googletest-1.13.0.tar.gz")
  endif()

  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_Declare(googletest
    URL               ${googletest_URL}
    URL_HASH          ${googletest_HASH}
  )

  FetchContent_GetProperties(googletest)
  if(NOT googletest_POPULATED)
    message(STATUS "Downloading googletest")
    FetchContent_Populate(googletest)
  endif()
  message(STATUS "googletest is downloaded to ${googletest_SOURCE_DIR}")
  message(STATUS "googletest's binary dir is ${googletest_BINARY_DIR}")

  if(APPLE)
    set(CMAKE_MACOSX_RPATH ON) # to solve the following warning on macOS
  endif()
  #[==[
  -- Generating done
    Policy CMP0042 is not set: MACOSX_RPATH is enabled by default.  Run "cmake
    --help-policy CMP0042" for policy details.  Use the cmake_policy command to
    set the policy and suppress this warning.

    MACOSX_RPATH is not specified for the following targets:

      gmock
      gmock_main
      gtest
      gtest_main

  This warning is for project developers.  Use -Wno-dev to suppress it.
  ]==]

  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)

  target_include_directories(gtest
    INTERFACE
      ${googletest_SOURCE_DIR}/googletest/include
      ${googletest_SOURCE_DIR}/googlemock/include
  )
endfunction()

download_googltest()
