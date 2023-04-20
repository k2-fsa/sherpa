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
  target_compile_options(sherpa-offline-websocket-server PRIVATE -Wno-deprecated-declarations)
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
  target_compile_options(sherpa-offline-websocket-client PRIVATE -Wno-deprecated-declarations)
endif()

add_executable(sherpa-online-websocket-server
  online-websocket-server.cc
  online-websocket-server-impl.cc
)
target_link_libraries(sherpa-online-websocket-server sherpa_cpp_api)

if(NOT WIN32)
  target_link_libraries(sherpa-online-websocket-server -pthread)
  target_compile_options(sherpa-online-websocket-server PRIVATE -Wno-deprecated-declarations)
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
    target_compile_options(sherpa-online-websocket-client-microphone PRIVATE -Wno-deprecated-declarations)
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
    foreach(exe IN LISTS bins)
      target_link_libraries(${exe} "-Wl,-rpath,${p}/sherpa/lib")
      target_link_libraries(${exe} "-Wl,-rpath,${p}/../lib")
    endforeach()
  endforeach()

  foreach(exe IN LISTS bins)
    target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
  endforeach()
endif()

install(TARGETS ${bins}
  DESTINATION  bin
)
