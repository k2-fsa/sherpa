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

#ifndef SHERPA_CPP_API_WEBSOCKET_HTTP_SERVER_H_
#define SHERPA_CPP_API_WEBSOCKET_HTTP_SERVER_H_

#include <fstream>
#include <string>
#include <unordered_map>

namespace sherpa {

/** Read a text or a binary file.
 *
 * @param filename The file to read.
 * @return Return the file content in a string.
 */
static std::string ReadFile(const std::string &filename) {
  std::ifstream file(filename);

  std::string ans;
  file.seekg(0, std::ios::end);
  ans.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  ans.assign((std::istreambuf_iterator<char>(file)),
             std::istreambuf_iterator<char>());
  return ans;
}

static const char *kKnownFiles[] = {
    // Please sort it alphabetically
    "/css/bootstrap.min.css",
    "/css/bootstrap.min.css.map",
    "/index.html",
    "/js/bootstrap.min.js",
    "/js/bootstrap.min.js.map",
    "/js/jquery-3.6.0.min.js",
    "/js/offline_record.js",
    "/js/offline_record.js",
    "/js/popper.min.js",
    "/js/popper.min.js.map",
    "/js/streaming_record.js",
    "/js/upload.js",
    "/k2-logo.png",
    "/nav-partial.html",
    "/offline_record.html",
    "/streaming_record.html",
    "/upload.html",
};

/** A very simple http server.
 *
 * It serves only static files, e.g., html, js., css, etc.
 */
class HttpServer {
 public:
  explicit HttpServer(const std::string &root) {
    for (const auto filename : kKnownFiles) {
      content_.emplace(filename, ReadFile(root + filename));
    }

    error_content_ = R"(
<!doctype html><html><head>
<title>Speech recognition with next-gen Kaldi</title><body>
<h1>404 ERROR! Please re-check your URL</h1>
</body></head></html>
    )";
  }

  /** Handle a request from the client.
   *
   * @param filename The filename the client is requesting.
   * @param content  On return, it contains the content of the file if found.
   *
   * @return Return true if the given file is found; return false otherwise.
   */
  bool ProcessRequest(const std::string &filename, std::string *content) const {
    auto it = content_.find(filename);
    if (it == content_.end()) {
      return false;
    }

    *content = it->second;
    return true;
  }

  /** Return a string for 404. */
  const std::string &GetErrorContent() const { return error_content_; }

 private:
  /**Return this string to the client for 404 page.*/
  std::string error_content_;

  /** Map filename to its content.*/
  std::unordered_map<std::string, std::string> content_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_HTTP_SERVER_H_
