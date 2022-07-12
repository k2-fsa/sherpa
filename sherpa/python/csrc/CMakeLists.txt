add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

# Please sort files alphabetically
pybind11_add_module(_sherpa
  rnnt_beam_search.cc
  rnnt_conformer_model.cc
  rnnt_conv_emformer_model.cc
  rnnt_emformer_model.cc
  rnnt_model.cc
  sherpa.cc
)

target_link_libraries(_sherpa PRIVATE sherpa_core)
if(UNIX AND NOT APPLE)
  target_link_libraries(_sherpa PRIVATE ${TORCH_DIR}/lib/libtorch_python.so)
  target_link_libraries(_sherpa PRIVATE ${PYTHON_LIBRARY})
elseif(WIN32)
  target_link_libraries(_sherpa PRIVATE ${TORCH_DIR}/lib/torch_python.lib)
  target_link_libraries(_sherpa PRIVATE ${PYTHON_LIBRARIES})
endif()
