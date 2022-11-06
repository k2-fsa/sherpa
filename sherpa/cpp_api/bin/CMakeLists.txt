add_executable(offline-recognizer offline-recognizer.cc)
target_link_libraries(offline-recognizer sherpa-cpp-api)
set_target_properties(offline-recognizer PROPERTIES OUTPUT_NAME "sherpa-offline-recognizer")

add_executable(test_online_recognizer test_online_recognizer.cc)
target_link_libraries(test_online_recognizer online_recognizer)

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
