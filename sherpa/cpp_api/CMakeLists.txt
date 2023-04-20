set(sherpa_cpp_api_srcs
  endpoint.cc
  fast-beam-search-config.cc
  feature-config.cc
  offline-recognizer.cc
  offline-stream.cc
  online-recognizer.cc
)
add_library(sherpa_cpp_api ${sherpa_cpp_api_srcs})
target_link_libraries(sherpa_cpp_api sherpa_core)

if(UNIT AND NOT APPLE)
  target_link_libraries(sherpa_cpp_api -pthread)
endif()

if(SHERPA_ENABLE_TESTS)
  add_executable(test-feature-config test-feature-config.cc)
  target_link_libraries(test-feature-config sherpa_cpp_api)

  add_executable(test-offline-stream test-offline-stream.cc)
  target_link_libraries(test-offline-stream sherpa_cpp_api)
endif()

file(MAKE_DIRECTORY
  ${PROJECT_BINARY_DIR}/include/sherpa/cpp_api
)

set(hdrs
  feature-config.h
  offline-recognizer.h
  offline-stream.h
  online-recognizer.h
  online-stream.h
  parse-options.h
)

file(COPY
  ${hdrs}
  DESTINATION
    ${PROJECT_BINARY_DIR}/include/sherpa/cpp_api
)

install(FILES ${hdrs}
  DESTINATION include/sherpa/cpp_api
)

install(FILES ${hdrs}
  DESTINATION ${PROJECT_BINARY_DIR}/include/sherpa/cpp_api
)

install(
  TARGETS sherpa_cpp_api
  DESTINATION lib
)

add_subdirectory(bin)

if(SHERPA_ENABLE_WEBSOCKET)
  add_subdirectory(websocket)
endif()
