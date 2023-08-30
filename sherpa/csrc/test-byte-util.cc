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

#include <string>

#include "gtest/gtest.h"
#include "sherpa/csrc/byte_util.h"
#include "sherpa/csrc/log.h"

namespace sherpa {

TEST(ByteUtil, TestBasic) {
  auto bu = GetByteUtil();
  std::string str = "Hello world";
  SHERPA_CHECK_EQ(bu->Decode(bu->Encode(str)), str);

  str = "世界人民大团结万岁";
  SHERPA_CHECK_EQ(bu->Decode(bu->Encode(str)), str);

  str = "美国 America vs China 中国 123 go!!!";
  SHERPA_CHECK_EQ(bu->Decode(bu->Encode(str)), str);
}

TEST(ByteUtil, TestInvalidBytes) {
  auto bu = GetByteUtil();
  std::string str = "ƍĩĴƎĩŗƋţŅƋ⁇Şœƌľţ";
  SHERPA_CHECK_EQ(bu->Decode(str), "我爱你中国");

  str = "ƍĩĴĩŗƋţŅƋŞœƌľţ";  // drop one byte in 爱
  SHERPA_CHECK_EQ(bu->Decode(str), "我你中国");

  str = "ƍĩƎĩŗƋţŅƋŞœƌľţ";  // drop one byte in 我
  SHERPA_CHECK_EQ(bu->Decode(str), "爱你中国");

  str = "ƍĩĴƎĩŗƋţŅƋŞœƌţ";  // drop one byte in 国
  SHERPA_CHECK_EQ(bu->Decode(str), "我爱你中");

  str = "ƍĩĴƎĩŗƋţŅƋœƌľ";  // drop one byte in 中 and 国
  SHERPA_CHECK_EQ(bu->Decode(str), "我爱你");

  str = "ƍĩĴƎĩŗƋţŅƋlœƌoľve";  // replace one byte in 中 and 国 with l o
  SHERPA_CHECK_EQ(bu->Decode(str), "我爱你love");
}

}  // namespace sherpa
