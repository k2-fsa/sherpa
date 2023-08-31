/**
 * Copyright      2023  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef SHERPA_CSRC_BYTE_UTIL_H_
#define SHERPA_CSRC_BYTE_UTIL_H_

#include <memory>
#include <string>
#include <vector>

namespace sherpa {

class ByteUtil;
using ByteUtilPtr = std::shared_ptr<ByteUtil>;

/* The class implements the functions in byte_utils.py
 * (https://github.com/k2-fsa/icefall/blob/master/icefall/byte_utils.py)
 * It will be used to decode the output hypothesis of model trained with byte
 * level bpe.
 *
 * Caution: The base characters (the byte token table) in the constructor MUST
 * be the same as the `PRINTABLE_BASE_CHARS` in icefall.
 */
class ByteUtil {
 public:
  ByteUtil();
  /*
   * Encode the normal string (for example, the transcripts in dataset) to a
   * special utf8 characters sequence, the characters are all in the byte2token_
   * table (see in the constructor). It breaks the non-ascii characters into
   * several characters (each byte a character), while the printable ascii will
   * keep the same.
   *
   * @param str  The original string.
   *
   * @returns  Returns the encoded string.
   */
  std::string Encode(const std::string &str) const;

  /* Decode the string encoded by Encode to its original one.
   * str should be equal to Decode(Encode(str)).
   *
   * Note: The str here actually represents a sequence of bytes, the number of
   * bytes equals to the number of utf8 characters, we will re-map this utf8
   * characters back to bytes with token2byte_ and then convert the bytes array
   * to a string. Sometimes, there will be some invalid bytes in the array, we
   * will drop these invalid bytes when decoding the bytes array. See more
   * examples in test-byte-util.cc.
   *
   * @returns Return the deocded string.
   */
  std::string Decode(const std::string &str) const;

 private:
  int32_t max_token_;                // The max token in byte2token_.
  std::vector<int16_t> token2byte_;  // map token to byte.
  std::vector<int16_t> byte2token_;  // map byte to token.

  /* Convert utf8 code points to corresponding character.
   * @param code  The utf8 code point.
   *
   * @return Returns the corresponding character (as std::string).
   */
  std::string CodePointToUTF8String(int32_t code) const;

  /* Convert bytes to corresponding utf8 code points.
   *
   * Note: We will skip invalid bytes (i.e the bytes can not combine into a
   * valid utf8 character).
   *
   * @param bytes  The pointer to the bytes array.
   * @param length  The length of bytes array.
   * @param code  The utf8 code points will be written here.
   */
  void BytesToCodePoints(const uint8_t *bytes, int32_t length,
                         std::vector<int32_t> *codes) const;
  /*
   * The utf8 string here is expected to be the encoded string (the string
   * encoded by Encode or the recognition result from a asr system built with
   * byte level bpe.
   *
   * This function first extract the utf8 characters from the str, then map them
   * to byte with token2byte_.
   *
   * @param str  The input string.
   * @param bytes  The converted bytes will be written here.
   */
  void UTF8StringToTokensAndMapToBytes(const std::string &str,
                                       std::vector<uint8_t> *bytes) const;
};

/*
 * Get the ByteUtil pointer, this guarantees the ByteUtil object only be
 * initialized once.
 */
const ByteUtilPtr GetByteUtil();

}  // namespace sherpa

#endif  // SHERPA_CSRC_BYTE_UTIL_H_
