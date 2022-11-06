add_executable(offline-recognizer offline-recognizer.cc)
target_link_libraries(offline-recognizer sherpa-cpp-api)
set_target_properties(offline-recognizer PROPERTIES OUTPUT_NAME "sherpa-offline-recognizer")

if(NOT WIN32)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(offline-recognizer "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")
  target_link_libraries(offline-recognizer "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
endif()

install(
  TARGETS offline-recognizer
  DESTINATION bin
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
