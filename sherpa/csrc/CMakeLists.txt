# Please sort the filenames alphabetically
set(sherpa_srcs
  base64-decode.cc
  byte_util.cc
  context-graph.cc
  fbank-features.cc
  file-utils.cc
  hypothesis.cc
  log.cc
  offline-conformer-ctc-model.cc
  offline-conformer-transducer-model.cc
  offline-ctc-greedy-search-decoder.cc
  offline-ctc-one-best-decoder.cc
  offline-model-config.cc
  offline-nemo-enc-dec-ctc-model-bpe.cc
  offline-sense-voice-model-config.cc
  offline-sense-voice-model-meta-data.cc
  offline-sense-voice-model.cc
  offline-stream.cc
  offline-transducer-fast-beam-search-decoder.cc
  offline-transducer-greedy-search-decoder.cc
  offline-transducer-modified-beam-search-decoder.cc
  offline-wav2vec2-ctc-model.cc
  offline-wenet-conformer-ctc-model.cc
  offline-whisper-model-config.cc
  offline-whisper-model-meta-data.cc
  offline-whisper-model.cc
  online-conformer-transducer-model.cc
  online-conv-emformer-transducer-model.cc
  online-emformer-transducer-model.cc
  online-lstm-transducer-model.cc
  online-stream.cc
  online-transducer-fast-beam-search-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-modified-beam-search-decoder.cc
  online-zipformer-transducer-model.cc
  online-zipformer2-transducer-model.cc
  parse-options.cc
  resample.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  symbol-table.cc
  text-utils.cc
  vad-model-config.cc
  voice-activity-detector-impl.cc
  voice-activity-detector.cc
  #
  speaker-embedding-extractor-model.cc
  speaker-embedding-extractor.cc
  speaker-embedding-extractor-impl.cc
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
    # test-offline-conformer-transducer-model.cc
    # test-online-conv-emformer-transducer-model.cc

    test-byte-util.cc
    test-context-graph.cc
    test-hypothesis.cc
    test-log.cc
    test-online-stream.cc
    test-parse-options.cc
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

configure_file(version.h.in ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)
message(STATUS "Generated ${CMAKE_CURRENT_BINARY_DIR}/version.h")
# sherpa-version does not have any dependencies. Its purpose is to help us debug
# issues.
add_executable(sherpa-version version.cc)
target_include_directories(sherpa-version PRIVATE ${CMAKE_BINARY_DIR})

add_executable(sherpa-vad sherpa-vad.cc)
target_link_libraries(sherpa-vad sherpa_core)

add_executable(sherpa-compute-speaker-similarity sherpa-compute-speaker-similarity.cc)
target_link_libraries(sherpa-compute-speaker-similarity sherpa_core)

install(TARGETS
    sherpa_core
  DESTINATION lib
)

install(
  TARGETS
   sherpa-version
   sherpa-vad
   sherpa-compute-speaker-similarity
  DESTINATION  bin
)
