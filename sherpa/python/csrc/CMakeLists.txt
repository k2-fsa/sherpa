add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

# Please sort files alphabetically
pybind11_add_module(_sherpa
  hypothesis.cc
  rnnt_beam_search.cc
  rnnt_conformer_model.cc
  rnnt_conv_emformer_model.cc
  rnnt_emformer_model.cc
  rnnt_model.cc
  sherpa.cc
)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_sherpa PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_sherpa PRIVATE "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/sherpa/lib")
endif()


target_link_libraries(_sherpa PRIVATE sherpa_core)
target_include_directories(_sherpa PRIVATE ${CMAKE_BINARY_DIR})

if(UNIX AND NOT APPLE)
  target_link_libraries(_sherpa PRIVATE ${TORCH_DIR}/lib/libtorch_python.so)
  target_link_libraries(_sherpa PRIVATE ${PYTHON_LIBRARY})
elseif(WIN32)
  target_link_libraries(_sherpa PRIVATE ${TORCH_DIR}/lib/torch_python.lib)
  target_link_libraries(_sherpa PRIVATE ${PYTHON_LIBRARIES})
endif()

install(TARGETS _sherpa
  DESTINATION ../
)
