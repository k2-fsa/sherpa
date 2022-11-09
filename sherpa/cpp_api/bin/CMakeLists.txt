add_executable(sherpa-offline offline-recognizer.cc)
target_link_libraries(sherpa-offline sherpa_cpp_api)

add_executable(sherpa-online online-recognizer.cc)
target_link_libraries(sherpa-online sherpa_cpp_api)

if(SHERPA_ENABLE_PORTAUDIO)
  add_executable(sherpa-online-microphone online-recognizer-microphone.cc)
  target_link_libraries(sherpa-online-microphone sherpa_cpp_api)
  if(BUILD_SHARED_LIBS)
    target_link_libraries(sherpa-online-microphone portaudio)
  else()
    target_link_libraries(sherpa-online-microphone portaudio_static)
  endif()
endif()

if(NOT WIN32)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(sherpa-offline "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")
  target_link_libraries(sherpa-offline "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")

  target_link_libraries(sherpa-online "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")
  target_link_libraries(sherpa-online "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")

  if(SHERPA_ENABLE_PORTAUDIO)
    target_link_libraries(sherpa-online-microphone "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")
    target_link_libraries(sherpa-online-microphone "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
  endif()
endif()

install(
  TARGETS sherpa-offline sherpa-online sherpa-online-microphone
  DESTINATION bin
)

if(SHERPA_ENABLE_PORTAUDIO)
  install(
    TARGETS sherpa-online-microphone
    DESTINATION bin
  )
endif()
