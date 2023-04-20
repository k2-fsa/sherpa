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

set(exe_list
  sherpa-offline
  sherpa-online
)

if(SHERPA_ENABLE_PORTAUDIO)
  list(APPEND exe_list sherpa-online-microphone)
endif()

if(NOT WIN32)
  if(NOT DEFINED ENV{VIRTUAL_ENV})
    message(STATUS "Outside a virtual environment")
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import site; print(';'.join(site.getsitepackages()))"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE path_list
    )
  else()
    message(STATUS "Inside a virtual environment")
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
    )
    set(path_list ${PYTHON_SITE_PACKAGE_DIR})
  endif()

  message(STATUS "path list: ${path_list}")
  foreach(p IN LISTS path_list)
    foreach(exe IN LISTS exe_list)
      target_link_libraries(${exe} "-Wl,-rpath,${p}/sherpa/lib")
      target_link_libraries(${exe} "-Wl,-rpath,${p}/../lib")
    endforeach()
  endforeach()

  foreach(exe IN LISTS exe_list)
    target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
  endforeach()
endif()

install(
  TARGETS ${exe_list}
  DESTINATION bin
)
