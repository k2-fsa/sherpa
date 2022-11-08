add_definitions(-DASIO_STANDALONE)
add_definitions(-D_WEBSOCKETPP_CPP11_STL_)

add_executable(sherpa-offline-websocket-server
  offline-websocket-server.cc
  offline-websocket-server-impl.cc
)
target_link_libraries(sherpa-offline-websocket-server
  sherpa_cpp_api
)
if(NOT WIN32)
  target_link_libraries(sherpa-offline-websocket-server -pthread)
endif()

add_executable(sherpa-offline-websocket-client
  offline-websocket-client.cc
)

target_link_libraries(sherpa-offline-websocket-client
  sherpa_core
  kaldi_native_io_core
)

if(NOT WIN32)
  target_link_libraries(sherpa-offline-websocket-client -pthread)
endif()

add_executable(sherpa-online-websocket-server
  online-websocket-server.cc
  online-websocket-server-impl.cc
)
target_link_libraries(sherpa-online-websocket-server sherpa_cpp_api)

if(NOT WIN32)
  target_link_libraries(sherpa-online-websocket-server -pthread)
endif()

add_executable(sherpa-online-websocket-client
  online-websocket-client.cc
)

target_link_libraries(sherpa-online-websocket-client
  sherpa_core
  kaldi_native_io_core
)

if(NOT WIN32)
  target_link_libraries(sherpa-online-websocket-client -pthread)
endif()

if(SHERPA_ENABLE_PORTAUDIO)
  add_executable(sherpa-online-websocket-client-microphone
    online-websocket-client-from-microphone.cc
    microphone.cc
  )

  target_link_libraries(sherpa-online-websocket-client-microphone sherpa_core)

  if(BUILD_SHARED_LIBS)
    target_link_libraries(sherpa-online-websocket-client-microphone portaudio)
  else()
    target_link_libraries(sherpa-online-websocket-client-microphone portaudio_static)
  endif()

  if(NOT WIN32)
    target_link_libraries(sherpa-online-websocket-client-microphone -pthread)
  endif()
endif()

set(bins
  sherpa-offline-websocket-server
  sherpa-offline-websocket-client
  sherpa-online-websocket-server
  sherpa-online-websocket-client
)
if(SHERPA_ENABLE_PORTAUDIO)
  list(APPEND bins sherpa-online-websocket-client-microphone)
endif()

foreach(exe IN LISTS bins)
  if(NOT WIN32)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
    )
    message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
    target_link_libraries(${exe} "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")
    target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
  endif()

  install(TARGETS
      ${exe}
    DESTINATION  bin
  )
endforeach()
