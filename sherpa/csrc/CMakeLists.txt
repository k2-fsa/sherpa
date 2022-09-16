# Please sort the filenames alphabetically
set(sherpa_srcs
  fbank_features.cc
  file_utils.cc
  hypothesis.cc
  log.cc
  offline_asr.cc
  parse_options.cc
  rnnt_beam_search.cc
  rnnt_conformer_model.cc
  rnnt_conv_emformer_model.cc
  rnnt_emformer_model.cc
  symbol_table.cc
)

add_library(sherpa_core ${sherpa_srcs})

if(SHERPA_HAVE_EXECINFO_H)
  target_compile_definitions(sherpa_core PRIVATE SHERPA_HAVE_EXECINFO_H=1)
endif()

if(SHERPA_HAVE_CXXABI_H)
  target_compile_definitions(sherpa_core PRIVATE SHERPA_HAVE_CXXABI_H=1)
endif()

target_link_libraries(sherpa_core PUBLIC
  ${TORCH_LIBRARIES}
  ${K2_LIBRARIES}
  ${KALDIFEAT_LIBRARIES}
  kaldi_native_io_core
  )

target_compile_definitions(sherpa_core PUBLIC SHERPA_TORCH_VERSION_MAJOR=${SHERPA_TORCH_VERSION_MAJOR})
target_compile_definitions(sherpa_core PUBLIC SHERPA_TORCH_VERSION_MINOR=${SHERPA_TORCH_VERSION_MINOR})
if(NOT WIN32)
  target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/k2/lib")
  target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/k2/lib64")
  target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/torch/lib")
  target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/torch/lib64")
endif()

if(DEFINED ENV{CONDA_PREFIX} AND APPLE)
  target_link_libraries(sherpa_core PUBLIC "-L $ENV{CONDA_PREFIX}/lib")
  target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,$ENV{CONDA_PREFIX}/lib")
endif()

if(SHERPA_ENABLE_TESTS)
  set(sherpa_test_srcs
    test_hypothesis.cc
    test_log.cc
    test_parse_options.cc
  )

  function(sherpa_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name ${name})
    add_executable(${target_name} "${source}")

    target_link_libraries(${target_name}
      PRIVATE
        sherpa_core
        gtest
        gtest_main
    )

    # NOTE: We set the working directory here so that
    # it works also on windows. The reason is that
    # the required DLLs are inside ${TORCH_DIR}/lib
    # and they can be found by the exe if the current
    # working directory is ${TORCH_DIR}\lib
    add_test(NAME "Test.${target_name}"
      COMMAND
      $<TARGET_FILE:${target_name}>
      WORKING_DIRECTORY ${TORCH_DIR}/lib
    )
  endfunction()

  foreach(source IN LISTS sherpa_test_srcs)
    sherpa_add_test(${source})
  endforeach()
endif()

add_executable(sherpa sherpa.cc)
target_link_libraries(sherpa sherpa_core)

if(NOT WIN32)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(sherpa "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}/sherpa/lib")

  target_link_libraries(sherpa "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/../lib")
endif()


configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)
message(STATUS "Generated ${CMAKE_CURRENT_BINARY_DIR}/version.h")
# sherpa-version does not have any dependencies. Its purpose is to help us debug
# issues.
add_executable(sherpa-version version.cc)
target_include_directories(sherpa-version PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS
    sherpa_core kaldi_native_io_core
  DESTINATION lib
)

install(TARGETS sherpa sherpa-version
  DESTINATION  bin
)
