/** Copyright      2023  Xiaomi Corporation (authors: Wei Kang)
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

#include "sherpa/csrc/byte_util.h"

#include <mutex>  // NOLINT
#include <string>

#include "sherpa/csrc/log.h"

namespace sherpa {

ByteUtil::ByteUtil() {
  // The table below is copied from
  // https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py
  // which is used to train byte level bpe, if you change the table in icefall
  // you have to change the table below accordingly.
  byte2token_ = std::vector<int16_t>(
      {256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269,
       270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
       284, 285, 286, 287, 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
       42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
       56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
       70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
       84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
       98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
       126, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300,
       301, 302, 303, 304, 305, 308, 309, 310, 311, 312, 313, 314, 315, 316,
       317, 318, 321, 322, 323, 324, 325, 326, 327, 328, 330, 331, 332, 333,
       334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347,
       348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361,
       362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375,
       376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390,
       391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404,
       405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
       419, 420, 421, 422});
  max_token_ = 422;  // the max number in above table
  token2byte_ =
      std::vector<int16_t>(max_token_ + 1, -1);  // the max token in byte2token_
                                                 // is 422, so we set the length
                                                 // of token2bytes_ 423.
  for (size_t i = 0; i < byte2token_.size(); ++i) {
    token2byte_[byte2token_[i]] = i;
  }
}

std::string ByteUtil::Encode(const std::string &str) const {
  std::ostringstream oss;
  const uint8_t *p = reinterpret_cast<const uint8_t *>(str.data());
  for (size_t i = 0; i < str.size(); ++i) {
    oss << CodePointToUTF8String(byte2token_[p[i]]);
  }
  return oss.str();
}

std::string ByteUtil::Decode(const std::string &str) const {
  std::vector<uint8_t> bytes;
  UTF8StringToTokensAndMapToBytes(str, &bytes);
  std::vector<int32_t> codes;
  BytesToCodePoints(bytes.data(), bytes.size(), &codes);
  std::ostringstream oss;
  for (size_t i = 0; i < codes.size(); ++i) {
    oss << CodePointToUTF8String(codes[i]);
  }
  return oss.str();
}

void ByteUtil::UTF8StringToTokensAndMapToBytes(
    const std::string &str, std::vector<uint8_t> *bytes) const {
  const char *data = str.data();
  bytes->clear();
  const size_t length = str.size();
  for (size_t i = 0; i < length; /* no update */) {
    int32_t c = data[i++] & 0xff;
    if ((c & 0x80) == 0) {
      if (c > max_token_ || token2byte_[c] == -1) {
        SHERPA_LOG(WARNING) << "Skip OOV token, code point : " << c
                            << " utf8 char : " << CodePointToUTF8String(c);
        continue;
      }
      bytes->push_back(token2byte_[c]);
    } else {
      if ((c & 0xc0) == 0x80) {
        SHERPA_LOG(FATAL) << "Invalid utf8 string : " << str
                          << ", code point : " << c;
      }
      int32_t count =
          (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) + (c >= 0xfc);
      int32_t code = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (i == length) {
          SHERPA_LOG(FATAL)
              << "Invalid utf8 string : " << str << ", code point : " << code;
        }
        char cb = data[i++];
        if ((cb & 0xc0) != 0x80) {
          SHERPA_LOG(FATAL)
              << "Invalid utf8 string : " << str << ", code point : " << code;
        }
        code = (code << 6) | (cb & 0x3f);
        count--;
      }
      if (code < 0) {
        // This should not be able to happen.
        SHERPA_LOG(FATAL) << "Invalid utf8 string : " << str
                          << ", code point : " << code;
      }
      if (code > max_token_ || token2byte_[code] == -1) {
        SHERPA_LOG(WARNING) << "Skip OOV token, code point : " << code
                            << " utf8 char : " << CodePointToUTF8String(code);
        continue;
      }
      bytes->push_back(token2byte_[code]);
    }
  }
}

void ByteUtil::BytesToCodePoints(const uint8_t *bytes, int32_t length,
                                 std::vector<int32_t> *codes) const {
  if (length <= 0) {
    return;
  }
  const char *data = reinterpret_cast<const char *>(bytes);
  int32_t idx = 1;  // means starting from the next byte
  for (int32_t i = 0; i < length; /* no update */) {
    int32_t c = data[i++] & 0xff;
    if ((c & 0x80) == 0) {
      codes->push_back(c);
      idx = i + 1;
    } else {
      if ((c & 0xc0) == 0x80) {
        BytesToCodePoints(bytes + idx, length - idx, codes);
        return;
      }
      int32_t count =
          (c >= 0xc0) + (c >= 0xe0) + (c >= 0xf0) + (c >= 0xf8) + (c >= 0xfc);
      int32_t code = c & ((1 << (6 - count)) - 1);
      while (count != 0) {
        if (i == length) {
          BytesToCodePoints(bytes + idx, length - idx, codes);
          return;
        }
        char cb = data[i++];
        if ((cb & 0xc0) != 0x80) {
          BytesToCodePoints(bytes + idx, length - idx, codes);
          return;
        }
        code = (code << 6) | (cb & 0x3f);
        count--;
      }
      if (code < 0) {
        BytesToCodePoints(bytes + idx, length - idx, codes);
        return;
      }
      codes->push_back(code);
      idx = i + 1;
    }
  }
}

std::string ByteUtil::CodePointToUTF8String(int32_t code) const {
  std::ostringstream ostr;
  if (code < 0) {
    SHERPA_LOG(FATAL) << "Invalid utf8 code point : " << code;
    return ostr.str();  // Unreachable code.
  } else if (code < 0x80) {
    ostr << static_cast<char>(code);
  } else if (code < 0x800) {
    ostr << static_cast<char>((code >> 6) | 0xc0);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x10000) {
    ostr << static_cast<char>((code >> 12) | 0xe0);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x200000) {
    ostr << static_cast<char>((code >> 18) | 0xf0);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else if (code < 0x4000000) {
    ostr << static_cast<char>((code >> 24) | 0xf8);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  } else {
    ostr << static_cast<char>((code >> 30) | 0xfc);
    ostr << static_cast<char>(((code >> 24) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 18) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 12) & 0x3f) | 0x80);
    ostr << static_cast<char>(((code >> 6) & 0x3f) | 0x80);
    ostr << static_cast<char>((code & 0x3f) | 0x80);
  }
  return ostr.str();
}

const ByteUtilPtr GetByteUtil() {
  static ByteUtilPtr bu = nullptr;
  static std::once_flag init_flag;

  std::call_once(init_flag,
                 []() { bu = std::make_shared<ByteUtil>(ByteUtil()); });
  SHERPA_CHECK_NE(bu, nullptr);
  return bu;
}

}  // namespace sherpa
