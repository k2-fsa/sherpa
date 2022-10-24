add_definitions(-DASIO_STANDALONE)
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
