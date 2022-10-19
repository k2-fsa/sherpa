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

struct FileMetadata {
  std::string filename;
  std::string mime_type;
};

struct FileContent {
  std::string content;
  std::string mime_type;
  FileContent() = default;
  FileContent(const std::string &filename, const std::string &mime_type)
      : content(ReadFile(filename)), mime_type(mime_type) {}
};

static const FileMetadata kKnownFiles[] = {
    // filename, content-type
    {"/index.html", "text/html"},
    {"/upload.html", "text/html"},
    {"/streaming_record.html", "text/html"},
    {"/offline_record.html", "text/html"},
    {"/js/jquery-3.6.0.min.js", "application/javascript"},
    {"/js/popper.min.js", "application/javascript"},
    {"/js/popper.min.js.map", "text/plain"},
    {"/js/bootstrap.min.js", "application/javascript"},
    {"/js/offline_record.js", "application/javascript"},
    {"/js/streaming_record.js", "application/javascript"},
    {"/js/offline_record.js", "application/javascript"},
    {"/js/upload.js", "application/javascript"},
    {"/js/bootstrap.min.js.map", "text/plain"},
    {"/css/bootstrap.min.css", "text/css"},
    {"/css/bootstrap.min.css.map", "text/plain"},
    {"/nav-partial.html", "text/html"},
    {"/k2-logo.png", "image/png"},
};

class HttpServer {
 public:
  explicit HttpServer(const std::string &root) {
    for (const auto &f : kKnownFiles) {
      auto name = root + f.filename;
      content_.emplace(f.filename, FileContent(name, f.mime_type));
    }

    error_content_ = R"(
<!doctype html><html><head>
<title>Speech recognition with next-gen Kaldi</title><body>
<h1>404 ERROR! Please re-check your URL</h1>
</body></head></html>
    )";
  }

  bool ProcessRequest(const std::string &filename, std::string *content,
                      std::string *mime_type) const {
    auto it = content_.find(filename);
    if (it == content_.end()) {
      return false;
    }

    *content = it->second.content;
    *mime_type = it->second.mime_type;
    return true;
  }

  const std::string &GetErrorContent() const { return error_content_; }

 private:
  std::string error_content_;
  std::unordered_map<std::string, FileContent> content_;
};

}  // namespace sherpa

#endif  // SHERPA_CPP_API_WEBSOCKET_HTTP_SERVER_H_
