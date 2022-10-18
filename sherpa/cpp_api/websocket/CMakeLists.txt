add_executable(websocket_server websocket_server.cc)
if(NOT WIN32)
  target_link_libraries(websocket_server -pthread)
endif()
