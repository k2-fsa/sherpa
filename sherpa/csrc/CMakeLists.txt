# Please sort the filenames alphabetically
set(sherpa_srcs
  rnnt_beam_search.cc
  rnnt_model.cc
)

add_library(sherpa_core ${sherpa_srcs})
target_link_libraries(sherpa_core PUBLIC ${TORCH_LIBRARIES})
