add_executable(websocket_server
  websocket_server.cc
  offline_websocket_server.cc
)
target_link_libraries(websocket_server
  offline_recognizer
)
if(NOT WIN32)
  target_link_libraries(websocket_server -pthread)
endif()
