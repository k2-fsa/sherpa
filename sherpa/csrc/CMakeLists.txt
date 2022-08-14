# Please sort the filenames alphabetically
set(sherpa_srcs
  file_utils.cc
  hypothesis.cc
  log.cc
  parse_options.cc
  rnnt_beam_search.cc
  rnnt_conformer_model.cc
  rnnt_conv_emformer_model.cc
  rnnt_emformer_model.cc
)

add_library(sherpa_core ${sherpa_srcs})

if(SHERPA_HAVE_EXECINFO_H)
  target_compile_definitions(sherpa_core PRIVATE SHERPA_HAVE_EXECINFO_H=1)
endif()

if(SHERPA_HAVE_CXXABI_H)
  target_compile_definitions(sherpa_core PRIVATE SHERPA_HAVE_CXXABI_H=1)
endif()

target_link_libraries(sherpa_core PUBLIC ${TORCH_LIBRARIES} ${K2_LIBRARIES})

target_compile_definitions(sherpa_core PUBLIC SHERPA_TORCH_VERSION_MAJOR=${SHERPA_TORCH_VERSION_MAJOR})
target_compile_definitions(sherpa_core PUBLIC SHERPA_TORCH_VERSION_MINOR=${SHERPA_TORCH_VERSION_MINOR})
target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/k2/lib")
target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/k2/lib64")
target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/torch/lib")
target_link_libraries(sherpa_core PUBLIC "-Wl,-rpath,${SHERPA_RPATH_ORIGIN}/torch/lib64")

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
target_link_libraries(sherpa
  sherpa_core
  ${KALDIFEAT_LIBRARIES}
  ${KALDI_NATIVE_IO_LIBRARIES}
)
