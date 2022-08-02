link_libraries(log libc++_shared.so android)

set(bin_dep_libs
  ${TORCH_LIBRARIES}
  sherpa_core
)

add_library(k2_device SHARED jni/k2.cc)
set_property(TARGET k2_device PROPERTY CXX_STANDARD 14)
target_link_libraries(k2_device ${bin_dep_libs})
