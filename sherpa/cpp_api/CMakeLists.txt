add_library(offline_recognizer offline_recognizer.cc)
target_link_libraries(offline_recognizer sherpa_core)
set_target_properties(offline_recognizer PROPERTIES OUTPUT_NAME "sherpa_offline_recognizer")

add_executable(test_decode_files test_decode_files.cc)
target_link_libraries(test_decode_files offline_recognizer)


# We use kaldi_native_io to read *.wav files, so we link to ${KALDI_NATIVE_IO_LIBRARIES} below
add_executable(test_decode_samples test_decode_samples.cc)
target_link_libraries(test_decode_samples offline_recognizer ${KALDI_NATIVE_IO_LIBRARIES})

# We use kaldi_native_io to read *.wav files, so we link to ${KALDI_NATIVE_IO_LIBRARIES} below
# Also, we use kaldifeat to compute fbank features, so we link to ${KALDIFEAT_LIBRARIES} below
add_executable(test_decode_features test_decode_features.cc)
target_link_libraries(test_decode_features
  offline_recognizer
  ${KALDI_NATIVE_IO_LIBRARIES}
  ${KALDIFEAT_LIBRARIES}
)

add_library(online_recognizer online_recognizer.cc)
target_link_libraries(online_recognizer sherpa_core)
set_target_properties(online_recognizer PROPERTIES OUTPUT_NAME "sherpa_online_recognizer")

add_executable(test_online_recognizer test_online_recognizer.cc)
target_link_libraries(test_online_recognizer online_recognizer)

if (SHERPA_ENABLE_WEBSOCKET_SERVER)
  add_subdirectory(websocket)
endif()

if(SHERPA_ENABLE_PORTAUDIO)
  add_executable(test_online_recognizer_microphone
    test_online_recognizer_microphone.cc
    )
  target_link_libraries(test_online_recognizer_microphone online_recognizer)
  if(BUILD_SHARED_LIBS)
    target_link_libraries(test_online_recognizer_microphone portaudio)
  else()
    target_link_libraries(test_online_recognizer_microphone portaudio_static)
  endif()
endif()

file(MAKE_DIRECTORY
  DESTINATION
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
