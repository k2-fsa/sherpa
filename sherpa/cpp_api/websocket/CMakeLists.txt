add_executable(websocket-client client.cc)
add_executable(websocket-server server.cc)
target_link_libraries(websocket-client PUBLIC sherpa_core)
target_link_libraries(websocket-server PUBLIC sherpa_core)
