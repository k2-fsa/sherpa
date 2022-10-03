function(download_boost)
  include(FetchContent)

  set(boost_URL   "https://boostorg.jfrog.io/artifactory/main/release/1.75.0/source/boost_1_75_0.tar.bz2")
  set(boost_HASH  "SHA256=953db31e016db7bb207f11432bef7df100516eeb746843fa0486a222e3fd49cb")

  FetchContent_Declare(boost
    URL         ${boost_URL}
    URL_HASH    ${boost_HASH}
  )

  FetchContent_GetProperties(boost)
  if(NOT boost_POPULATED)
    message(STATUS "Downloading boost ${boost_URL} (May take 10 minutes)")
    FetchContent_Populate(boost)
  endif()
  message(STATUS "boost is downloaded to ${boost_SOURCE_DIR}")
  # add_subdirectory(${boost_SOURCE_DIR} ${boost_BINARY_DIR} EXCLUDE_FROM_ALL)

  include_directories(${boost_SOURCE_DIR})

  if(MSVC)
    add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_ALL_NO_LIB)
  endif()
endfunction()

download_boost()
