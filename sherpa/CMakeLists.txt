add_subdirectory(csrc)
add_subdirectory(python)

add_subdirectory(cpp_api)

install(DIRECTORY bin/ DESTINATION bin/)
