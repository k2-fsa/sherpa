add_definitions(-DASIO_STANDALONE)
add_definitions(-D_WEBSOCKETPP_CPP11_STL_)

add_executable(offline_websocket_server
  offline_websocket_server.cc
  offline_websocket_server_impl.cc
)
target_link_libraries(offline_websocket_server
  offline_recognizer
)
if(NOT WIN32)
  target_link_libraries(offline_websocket_server -pthread)
endif()

add_executable(offline_websocket_client
  offline_websocket_client.cc
)

target_link_libraries(offline_websocket_client
  sherpa_core
  kaldi_native_io_core
)

if(NOT WIN32)
  target_link_libraries(offline_websocket_client -pthread)
endif()

add_executable(online_websocket_server
  online_websocket_server.cc
  online_websocket_server_impl.cc
)
target_link_libraries(online_websocket_server
  online_recognizer
)

if(NOT WIN32)
  target_link_libraries(online_websocket_server -pthread)
endif()

add_executable(online_websocket_client
  online_websocket_client.cc
)

target_link_libraries(online_websocket_client
  sherpa_core
  kaldi_native_io_core
)

if(NOT WIN32)
  target_link_libraries(online_websocket_client -pthread)
endif()


if(SHERPA_ENABLE_PORTAUDIO)
  add_executable(online_websocket_client_from_microphone
    online_websocket_client_from_microphone.cc
    microphone.cc
  )

  target_link_libraries(online_websocket_client_from_microphone sherpa_core)

  if(BUILD_SHARED_LIBS)
    target_link_libraries(online_websocket_client_from_microphone portaudio)
  else()
    target_link_libraries(online_websocket_client_from_microphone portaudio_static)
  endif()

  if(NOT WIN32)
    target_link_libraries(online_websocket_client_from_microphone -pthread)
  endif()
endif()

set(bins
  offline_websocket_server
  offline_websocket_client
  online_websocket_server
  online_websocket_client
)
if(SHERPA_ENABLE_PORTAUDIO)
  list(APPEND bins online_websocket_client_from_microphone)
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
