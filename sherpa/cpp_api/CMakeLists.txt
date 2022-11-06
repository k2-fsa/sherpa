add_library(offline_recognizer offline_recognizer.cc)
target_link_libraries(offline_recognizer sherpa_core)
set_target_properties(offline_recognizer PROPERTIES OUTPUT_NAME "sherpa_offline_recognizer")

add_library(online_recognizer online_recognizer.cc)
target_link_libraries(online_recognizer sherpa_core)
set_target_properties(online_recognizer PROPERTIES OUTPUT_NAME "sherpa_online_recognizer")


file(MAKE_DIRECTORY
  ${PROJECT_BINARY_DIR}/include/sherpa/cpp_api
)

set(hdrs
  offline_recognizer.h
  online_recognizer.h
  online_stream.h
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
  TARGETS offline_recognizer online_recognizer
  DESTINATION lib
)

add_subdirectory(demo)

if(SHERPA_ENABLE_WEBSOCKET)
  add_subdirectory(websocket)
endif()
