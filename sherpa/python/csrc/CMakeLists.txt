add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

# Please sort files alphabetically
pybind11_add_module(_sherpa
  rnnt_beam_search.cc
  rnnt_model.cc
  sherpa.cc
)

target_link_libraries(_sherpa PRIVATE sherpa_core)
if(UNIX AND NOT APPLE)
  target_link_libraries(_sherpa PUBLIC ${TORCH_DIR}/lib/libtorch_python.so)
  target_link_libraries(_sherpa PUBLIC ${PYTHON_LIBRARY})
elseif(WIN32)
  target_link_libraries(_sherpa PUBLIC ${TORCH_DIR}/lib/torch_python.lib)
  target_link_libraries(_sherpa PUBLIC ${PYTHON_LIBRARIES})
endif()

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_sherpa PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()
