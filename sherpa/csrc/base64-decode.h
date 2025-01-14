// sherpa/csrc/base64-decode.h
//
// Copyright (c)  2022-2025  Xiaomi Corporation

#ifndef SHERPA_CSRC_BASE64_DECODE_H_
#define SHERPA_CSRC_BASE64_DECODE_H_

#include <string>

namespace sherpa {

/** @param s A base64 encoded string.
 *  @return Return the decoded string.
 */
std::string Base64Decode(const std::string &s);

}  // namespace sherpa

#endif  // SHERPA_CSRC_BASE64_DECODE_H_
