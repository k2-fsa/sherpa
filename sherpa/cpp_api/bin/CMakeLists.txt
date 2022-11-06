add_executable(test-decode-files test_decode_files.cc)
target_link_libraries(test-decode-files sherpa-cpp-api)

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
