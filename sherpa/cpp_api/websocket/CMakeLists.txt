add_executable(websocket-client client.cc)
add_executable(websocket-server server.cc)
target_link_libraries(websocket-client PRIVATE sherpa_core)
target_link_libraries(websocket-server PRIVATE sherpa_core)
