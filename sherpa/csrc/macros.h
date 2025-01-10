// sherpa/csrc/macros.h
//
// Copyright      2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_MACROS_H_
#define SHERPA_CSRC_MACROS_H_
#include <stdio.h>
#include <stdlib.h>

#include <utility>

#define SHERPA_LOGE(...)                             \
  do {                                               \
    fprintf(stderr, "%s:%s:%d ", __FILE__, __func__, \
            static_cast<int>(__LINE__));             \
    fprintf(stderr, ##__VA_ARGS__);                  \
    fprintf(stderr, "\n");                           \
  } while (0)

#define SHERPA_EXIT(code) exit(code)

#endif  // SHERPA_CSRC_MACROS_H_
