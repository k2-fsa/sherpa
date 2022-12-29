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

/*
 * Stack trace related stuff is from kaldi.
 * Refer to
 * https://github.com/kaldi-asr/kaldi/blob/master/src/base/kaldi-error.cc
 */

#include "sherpa/csrc/log.h"

#ifdef SHERPA_HAVE_EXECINFO_H
#include <execinfo.h>  // To get stack trace in error messages.
#ifdef SHERPA_HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // SHERPA_HAVE_CXXABI_H
#endif  // SHERPA_HAVE_EXECINFO_H
#include <stdlib.h>
#include <ctime>
#include <iomanip>
#include <string>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>

template<std::size_t V, std::size_t C = 0,
         typename std::enable_if<(V < 10), int>::type = 0>
constexpr std::size_t log10ish() {
    return C;
}

template<std::size_t V, std::size_t C = 0,
         typename std::enable_if<(V >= 10), int>::type = 0>
constexpr std::size_t log10ish() {
    return log10ish<V / 10, C + 1>();
}

// A class to support using different precisions, chrono clocks and formats
template<class Precision = std::chrono::seconds,
         class Clock = std::chrono::system_clock>
class log_watch {
public:
    // some convenience typedefs and "decimal_width" for sub second precisions
    using precision_type = Precision;
    using ratio_type = typename precision_type::period;
    using clock_type = Clock;
    static constexpr auto decimal_width = log10ish<ratio_type{}.den>();
    static_assert(ratio_type{}.num <= ratio_type{}.den,
                  "Only second or sub second precision supported");
    static_assert(ratio_type{}.num == 1, "Unsupported precision parameter");
    // default format: "%Y-%m-%dT%H:%M:%S"
    log_watch(const std::string& format = "%FT%T") : m_format(format) {}

    template<class P, class C>
    friend std::ostream& operator<<(std::ostream&, const log_watch<P, C>&);

private:
    std::string m_format;
};

template<class Precision, class Clock>
std::ostream& operator<<(std::ostream& os, const log_watch<Precision, Clock>& lw) {
    // get current system clock
    auto time_point = Clock::now();

    // extract std::time_t from time_point
    std::time_t t = Clock::to_time_t(time_point);

    // output the part supported by std::tm
    os << std::put_time(std::localtime(&t), lw.m_format.c_str());

    // only involve chrono duration calc for displaying sub second precisions
    if(lw.decimal_width) { // if constexpr( ... in C++17
        // get duration since epoch
        auto dur = time_point.time_since_epoch();

        // extract the sub second part from the duration since epoch
        auto ss =
            std::chrono::duration_cast<Precision>(dur) % std::chrono::seconds{1};

        // output the sub second part
        os << std::setfill('0') << std::setw(lw.decimal_width) << ss.count();
    }

    return os;
}

namespace sherpa {

std::string GetDateTimeStr() {
  log_watch<std::chrono::milliseconds> ms("%F %T.");
  std::ostringstream os;
  os << ms;  // yyyy-mm-dd hh:mm:ss
  return os.str();
}

static bool LocateSymbolRange(const std::string &trace_name, std::size_t *begin,
                              std::size_t *end) {
  // Find the first '_' with leading ' ' or '('.
  *begin = std::string::npos;
  for (std::size_t i = 1; i < trace_name.size(); ++i) {
    if (trace_name[i] != '_') {
      continue;
    }
    if (trace_name[i - 1] == ' ' || trace_name[i - 1] == '(') {
      *begin = i;
      break;
    }
  }
  if (*begin == std::string::npos) {
    return false;
  }
  *end = trace_name.find_first_of(" +", *begin);
  return *end != std::string::npos;
}

#ifdef SHERPA_HAVE_EXECINFO_H
static std::string Demangle(const std::string &trace_name) {
#ifndef SHERPA_HAVE_CXXABI_H
  return trace_name;
#else   // SHERPA_HAVE_CXXABI_H
  // Try demangle the symbol. We are trying to support the following formats
  // produced by different platforms:
  //
  // Linux:
  //   ./kaldi-error-test(_ZN5kaldi13UnitTestErrorEv+0xb) [0x804965d]
  //
  // Mac:
  //   0 server 0x000000010f67614d _ZNK5kaldi13MessageLogger10LogMessageEv + 813
  //
  // We want to extract the name e.g., '_ZN5kaldi13UnitTestErrorEv' and
  // demangle it info a readable name like kaldi::UnitTextError.
  std::size_t begin, end;
  if (!LocateSymbolRange(trace_name, &begin, &end)) {
    return trace_name;
  }
  std::string symbol = trace_name.substr(begin, end - begin);
  int status;
  char *demangled_name = abi::__cxa_demangle(symbol.c_str(), 0, 0, &status);
  if (status == 0 && demangled_name != nullptr) {
    symbol = demangled_name;
    free(demangled_name);
  }
  return trace_name.substr(0, begin) + symbol +
         trace_name.substr(end, std::string::npos);
#endif  // SHERPA_HAVE_CXXABI_H
}
#endif  // SHERPA_HAVE_EXECINFO_H

std::string GetStackTrace() {
  std::string ans;
#ifdef SHERPA_HAVE_EXECINFO_H
  constexpr const std::size_t kMaxTraceSize = 50;
  constexpr const std::size_t kMaxTracePrint = 50;  // Must be even.
                                                    // Buffer for the trace.
  void *trace[kMaxTraceSize];
  // Get the trace.
  std::size_t size = backtrace(trace, kMaxTraceSize);
  // Get the trace symbols.
  char **trace_symbol = backtrace_symbols(trace, size);
  if (trace_symbol == nullptr) return ans;

  // Compose a human-readable backtrace string.
  ans += "[ Stack-Trace: ]\n";
  if (size <= kMaxTracePrint) {
    for (std::size_t i = 0; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
  } else {  // Print out first+last (e.g.) 5.
    for (std::size_t i = 0; i < kMaxTracePrint / 2; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    ans += ".\n.\n.\n";
    for (std::size_t i = size - kMaxTracePrint / 2; i < size; ++i) {
      ans += Demangle(trace_symbol[i]) + "\n";
    }
    if (size == kMaxTraceSize)
      ans += ".\n.\n.\n";  // Stack was too long, probably a bug.
  }

  // We must free the array of pointers allocated by backtrace_symbols(),
  // but not the strings themselves.
  free(trace_symbol);
#endif  // SHERPA_HAVE_EXECINFO_H
  return ans;
}

}  // namespace sherpa
