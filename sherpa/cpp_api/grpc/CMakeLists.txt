add_definitions(-DASIO_STANDALONE)

# compile sherpo.proto
set(PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(PROTO_IN "${CMAKE_CURRENT_SOURCE_DIR}")
set(grpc_BINARY_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(grpc_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
include_directories(${CMAKE_BINARY_DIR})
add_custom_command(
  OUTPUT  ${PROTO_DIR}/sherpa.pb.cc
          ${PROTO_DIR}/sherpa.pb.h
          ${PROTO_DIR}/sherpa.grpc.pb.cc
          ${PROTO_DIR}/sherpa.grpc.pb.h
  COMMAND ${grpc_BINARY_DIR}/protoc
  ARGS --grpc_out "${PROTO_DIR}"
    --cpp_out "${PROTO_DIR}"
    -I "${PROTO_IN}"
    --plugin=protoc-gen-grpc=${grpc_BINARY_DIR}/grpc_cpp_plugin
    sherpa.proto)

add_executable(sherpa-online-grpc-server
  online-grpc-server.cc
  online-grpc-server-impl.cc
  ${PROTO_DIR}/sherpa.pb.cc
  ${PROTO_DIR}/sherpa.grpc.pb.cc
)
target_link_libraries(sherpa-online-grpc-server sherpa_cpp_api grpc++ grpc++_reflection)

if(NOT WIN32)
  target_link_libraries(sherpa-online-grpc-server -pthread)
  target_compile_options(sherpa-online-grpc-server PRIVATE -Wno-deprecated-declarations)
endif()

add_executable(sherpa-online-grpc-client
  online-grpc-client.cc
  online-grpc-client-impl.cc
  ${PROTO_DIR}/sherpa.pb.cc
  ${PROTO_DIR}/sherpa.grpc.pb.cc
)

target_link_libraries(sherpa-online-grpc-client
  sherpa_core
  kaldi_native_io_core
  grpc++
  grpc++_reflection
)

if(NOT WIN32)
  target_link_libraries(sherpa-online-grpc-client -pthread)
endif()

set(bins
  sherpa-online-grpc-server
  sherpa-online-grpc-client
)

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

  # add additional paths
  set(additional_paths
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/torch/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/torch/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/k2/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/k2/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/kaldifeat/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/kaldifeat/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sherpa/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/sherpa/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/torch/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/torch/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/k2/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/k2/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/kaldifeat/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/kaldifeat/lib64
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/sherpa/lib
    ${SHERPA_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/dist-packages/sherpa/lib64
    )
  message(STATUS "additional_paths: ${additional_paths}")
  foreach(p IN LISTS additional_paths)
    foreach(exe IN LISTS bins)
      target_link_libraries(${exe} "-Wl,-rpath,${p}")
      target_link_libraries(${exe} "-Wl,-rpath,${p}")
    endforeach()
  endforeach()
endif()

install(TARGETS ${bins}
  DESTINATION  bin
)
