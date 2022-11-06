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
