/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gtest/gtest.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

TEST(Log, TestLog) {
  SHERPA_LOG(TRACE) << "this is a trace message";
  SHERPA_LOG(DEBUG) << "this is a debug message";
  SHERPA_LOG(INFO) << "this is an info message";
  SHERPA_LOG(WARNING) << "this is a warning message";
  SHERPA_LOG(ERROR) << "this is an error message";

  ASSERT_THROW(SHERPA_LOG(FATAL) << "This will crash the program",
               std::runtime_error);

  // For debug build

  SHERPA_DLOG(TRACE) << "this is a trace message for debug build";
  SHERPA_DLOG(DEBUG) << "this is a trace message for debug build";
  SHERPA_DLOG(INFO) << "this is a trace message for debug build";
  SHERPA_DLOG(ERROR) << "this is an error message for debug build";
  SHERPA_DLOG(WARNING) << "this is a trace message for debug build";

#if !defined(NDEBUG)
  ASSERT_THROW(SHERPA_DLOG(FATAL) << "this is a trace message for debug build",
               std::runtime_error);
#endif
}

TEST(Log, TestCheck) {
  SHERPA_CHECK_EQ(1, 1) << "ok";
  SHERPA_CHECK_LE(1, 3) << "ok";
  SHERPA_CHECK_LT(1, 2) << "ok";
  SHERPA_CHECK_GT(2, 1) << "ok";
  SHERPA_CHECK_GE(2, 1) << "ok";

  ASSERT_THROW(SHERPA_CHECK_EQ(2, 1) << "bad things happened",
               std::runtime_error);

  // for debug build
  SHERPA_DCHECK_EQ(1, 1) << "ok";
  SHERPA_DCHECK_LE(1, 3) << "ok";
  SHERPA_DCHECK_LT(1, 2) << "ok";
  SHERPA_DCHECK_GT(2, 1) << "ok";
  SHERPA_DCHECK_GE(2, 1) << "ok";

#if !defined(NDEBUG)
  ASSERT_THROW(SHERPA_CHECK_EQ(2, 1) << "bad things happened",
               std::runtime_error);
#endif
}

}  // namespace sherpa
