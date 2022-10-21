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
