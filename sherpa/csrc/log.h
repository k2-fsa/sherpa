/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

// The content in this file is copied/modified from
// https://github.com/k2-fsa/k2/blob/master/k2/csrc/log.h
#ifndef SHERPA_CSRC_LOG_H_
#define SHERPA_CSRC_LOG_H_

#include <stdio.h>

#include <mutex>  // NOLINT
#include <sstream>
#include <string>

namespace sherpa {

#if defined(NDEBUG)
constexpr bool kDisableDebug = true;
#else
constexpr bool kDisableDebug = false;
#endif

enum class LogLevel {
  kTrace = 0,
  kDebug = 1,
  kInfo = 2,
  kWarning = 3,
  kError = 4,
  kFatal = 5,  // print message and abort the program
};

// They are used in SHERPA_LOG(xxx), so their names
// do not follow the google c++ code style
//
// You can use them in the following way:
//
//  SHERPA_LOG(TRACE) << "some message";
//  SHERPA_LOG(DEBUG) << "some message";
//
constexpr LogLevel TRACE = LogLevel::kTrace;
constexpr LogLevel DEBUG = LogLevel::kDebug;
constexpr LogLevel INFO = LogLevel::kInfo;
constexpr LogLevel WARNING = LogLevel::kWarning;
constexpr LogLevel ERROR = LogLevel::kError;
constexpr LogLevel FATAL = LogLevel::kFatal;

std::string GetStackTrace();

// Return a string with the format: yyyy-mm-dd hh:mm:ss
// For instance, 2022-08-15 11:44:44
std::string GetDateTimeStr();

/* Return the current log level.


   If the current log level is TRACE, then all logged messages are printed out.

   If the current log level is DEBUG, log messages with "TRACE" level are not
   shown and all other levels are printed out.

   Similarly, if the current log level is INFO, log message with "TRACE" and
   "DEBUG" are not shown and all other levels are printed out.

   If it is FATAL, then only FATAL messages are shown.
 */
inline LogLevel GetCurrentLogLevel() {
  static LogLevel log_level = INFO;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    const char *env_log_level = std::getenv("SHERPA_LOG_LEVEL");
    if (env_log_level == nullptr) return;

    std::string s = env_log_level;
    if (s == "TRACE")
      log_level = TRACE;
    else if (s == "DEBUG")
      log_level = DEBUG;
    else if (s == "INFO")
      log_level = INFO;
    else if (s == "WARNING")
      log_level = WARNING;
    else if (s == "ERROR")
      log_level = ERROR;
    else if (s == "FATAL")
      log_level = FATAL;
    else
      fprintf(stderr,
              "Unknown SHERPA_LOG_LEVEL: %s"
              "\nSupported values are: "
              "TRACE, DEBUG, INFO, WARNING, ERROR, FATAL",
              s.c_str());
  });
  return log_level;
}

inline bool EnableAbort() {
  static std::once_flag init_flag;
  static bool enable_abort = false;
  std::call_once(init_flag, []() {
    enable_abort = (std::getenv("SHERPA_ABORT") != nullptr);
  });
  return enable_abort;
}

class Logger {
 public:
  Logger(const char *filename, const char *func_name, uint32_t line_num,
         LogLevel level)
      : filename_(filename),
        func_name_(func_name),
        line_num_(line_num),
        level_(level) {
    cur_level_ = GetCurrentLogLevel();
    switch (level) {
      case TRACE:
        if (cur_level_ <= TRACE) fprintf(stderr, "[T] ");
        break;
      case DEBUG:
        if (cur_level_ <= DEBUG) fprintf(stderr, "[D] ");
        break;
      case INFO:
        if (cur_level_ <= INFO) fprintf(stderr, "[I] ");
        break;
      case WARNING:
        if (cur_level_ <= WARNING) fprintf(stderr, "[W] ");
        break;
      case ERROR:
        if (cur_level_ <= ERROR) fprintf(stderr, "[E] ");
        break;
      case FATAL:
        if (cur_level_ <= FATAL) fprintf(stderr, "[F] ");
        break;
    }

    if (cur_level_ <= level_) {
      fprintf(stderr, "%s:%u:%s %s ", filename, line_num, func_name,
              GetDateTimeStr().c_str());
    }
  }

  ~Logger() noexcept(false) {
    static constexpr const char *kErrMsg = R"(
    Some bad things happened. Please read the above error messages and stack
    trace. If you are using Python, the following command may be helpful:

      gdb --args python /path/to/your/code.py

    (You can use `gdb` to debug the code. Please consider compiling
    a debug version of sherpa.).

    If you are unable to fix it, please open an issue at:

      https://github.com/k2-fsa/sherpa/issues/new
    )";
    fprintf(stderr, "\n");
    if (level_ == FATAL) {
      std::string stack_trace = GetStackTrace();
      if (!stack_trace.empty()) {
        fprintf(stderr, "\n\n%s\n", stack_trace.c_str());
      }

      fflush(nullptr);

      if (EnableAbort()) {
        // NOTE: abort() will terminate the program immediately without
        // printing the Python stack backtrace.
        abort();
      }

      throw std::runtime_error(kErrMsg);
    }
  }

  const Logger &operator<<(bool b) const {
    if (cur_level_ <= level_) {
      fprintf(stderr, b ? "true" : "false");
    }
    return *this;
  }

  const Logger &operator<<(int8_t i) const {
    if (cur_level_ <= level_) fprintf(stderr, "%d", i);
    return *this;
  }

  const Logger &operator<<(const char *s) const {
    if (cur_level_ <= level_) fprintf(stderr, "%s", s);
    return *this;
  }

  const Logger &operator<<(int32_t i) const {
    if (cur_level_ <= level_) fprintf(stderr, "%d", i);
    return *this;
  }

  const Logger &operator<<(uint32_t i) const {
    if (cur_level_ <= level_) fprintf(stderr, "%u", i);
    return *this;
  }

  const Logger &operator<<(uint64_t i) const {
    if (cur_level_ <= level_)
      fprintf(stderr, "%llu", (long long unsigned int)i);  // NOLINT
    return *this;
  }

  const Logger &operator<<(int64_t i) const {
    if (cur_level_ <= level_)
      fprintf(stderr, "%lli", (long long int)i);  // NOLINT
    return *this;
  }

  const Logger &operator<<(float f) const {
    if (cur_level_ <= level_) fprintf(stderr, "%f", f);
    return *this;
  }

  const Logger &operator<<(double d) const {
    if (cur_level_ <= level_) fprintf(stderr, "%f", d);
    return *this;
  }

  template <typename T>
  const Logger &operator<<(const T &t) const {
    // require T overloads operator<<
    std::ostringstream os;
    os << t;
    return *this << os.str().c_str();
  }

  // specialization to fix compile error: `stringstream << nullptr` is ambiguous
  const Logger &operator<<(const std::nullptr_t &null) const {
    if (cur_level_ <= level_) *this << "(null)";
    return *this;
  }

 private:
  const char *filename_;
  const char *func_name_;
  uint32_t line_num_;
  LogLevel level_;
  LogLevel cur_level_;
};

class Voidifier {
 public:
  void operator&(const Logger &)const {}
};

}  // namespace sherpa

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__) || \
    defined(__PRETTY_FUNCTION__)
// for clang and GCC
#define SHERPA_FUNC __PRETTY_FUNCTION__
#else
// for other compilers
#define SHERPA_FUNC __func__
#endif

#define SHERPA_STATIC_ASSERT(x) static_assert(x, "")

#define SHERPA_CHECK(x)                                                        \
  (x) ? (void)0                                                                \
      : ::sherpa::Voidifier() &                                                \
            ::sherpa::Logger(__FILE__, SHERPA_FUNC, __LINE__, ::sherpa::FATAL) \
                << "Check failed: " << #x << " "

// WARNING: x and y may be evaluated multiple times, but this happens only
// when the check fails. Since the program aborts if it fails, we don't think
// the extra evaluation of x and y matters.
//
// CAUTION: we recommend the following use case:
//
//      auto x = Foo();
//      auto y = Bar();
//      SHERPA_CHECK_EQ(x, y) << "Some message";
//
//  And please avoid
//
//      SHERPA_CHECK_EQ(Foo(), Bar());
//
//  if `Foo()` or `Bar()` causes some side effects, e.g., changing some
//  local static variables or global variables.
#define _SHERPA_CHECK_OP(x, y, op)                                             \
  ((x)op(y))                                                                   \
      ? (void)0                                                                \
      : ::sherpa::Voidifier() &                                                \
            ::sherpa::Logger(__FILE__, SHERPA_FUNC, __LINE__, ::sherpa::FATAL) \
                << "Check failed: " << #x << " " << #op << " " << #y << " ("   \
                << (x) << " vs. " << (y) << ") "

#define SHERPA_CHECK_EQ(x, y) _SHERPA_CHECK_OP(x, y, ==)
#define SHERPA_CHECK_NE(x, y) _SHERPA_CHECK_OP(x, y, !=)
#define SHERPA_CHECK_LT(x, y) _SHERPA_CHECK_OP(x, y, <)
#define SHERPA_CHECK_LE(x, y) _SHERPA_CHECK_OP(x, y, <=)
#define SHERPA_CHECK_GT(x, y) _SHERPA_CHECK_OP(x, y, >)
#define SHERPA_CHECK_GE(x, y) _SHERPA_CHECK_OP(x, y, >=)

#define SHERPA_LOG(x) \
  ::sherpa::Logger(__FILE__, SHERPA_FUNC, __LINE__, ::sherpa::x)

// ------------------------------------------------------------
//       For debug check
// ------------------------------------------------------------
// If you define the macro "-D NDEBUG" while compiling sherpa, the following
// macros are in fact empty and does nothing.

#define SHERPA_DCHECK(x) ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK(x)

#define SHERPA_DCHECK_EQ(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_EQ(x, y)

#define SHERPA_DCHECK_NE(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_NE(x, y)

#define SHERPA_DCHECK_LT(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_LT(x, y)

#define SHERPA_DCHECK_LE(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_LE(x, y)

#define SHERPA_DCHECK_GT(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_GT(x, y)

#define SHERPA_DCHECK_GE(x, y) \
  ::sherpa::kDisableDebug ? (void)0 : SHERPA_CHECK_GE(x, y)

#define SHERPA_DLOG(x) \
  ::sherpa::kDisableDebug ? (void)0 : ::sherpa::Voidifier() & SHERPA_LOG(x)

#endif  // SHERPA_CSRC_LOG_H_
