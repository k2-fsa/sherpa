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

#include "kaldi_native_io/csrc/kaldi-table.h"
#include "kaldi_native_io/csrc/text-utils.h"
#include "kaldi_native_io/csrc/wave-reader.h"
#include "sherpa/cpp_api/online_stream.h"
#include "sherpa/csrc/fbank_features.h"
#include "sherpa/csrc/log.h"
#include "sherpa/csrc/online_asr.h"
#include "sherpa/csrc/parse_options.h"

static constexpr const char *kUsageMessage = R"(
Online (streaming) automatic speech recognition with sherpa.

Usage:
(1) View help information.

  ./bin/sherpa-online --help

(2) Use a pretrained model for recognition

  ./bin/sherpa-online \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    foo.wav \
    bar.wav

Note: You can get pre-trained models for testing by visiting
 - English: https://huggingface.co/Zengwei/icefall-asr-librispeech-conv-emformer-transducer-stateless2-2022-07-05

(3) Decode wav.scp

  ./bin/sherpa-online \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=false \
    --use-wav-scp=false \
    scp:wav.scp \
    ark,scp,t:results.ark,results.scp
)";

/** Decode a list of 1-d wave samples.
 *
 * @param online_asr  An instance of OnlineAsr.
 * @param samples  Each entry is a 1-D tensor containing audio samples.
 * @return Return the decoded results.
 */
static std::vector<std::string> DecodeWaves(
	sherpa::OnlineAsr &online_asr,  // NOLINT
	const std::vector<torch::Tensor> &samples) {
    using torch::indexing::Slice;
    float sample_rate = online_asr.Opts().fbank_opts.frame_opts.samp_freq;
    int32_t frame_size = 4096;
    int32_t batch_size = samples.size();
    std::vector<int> streams_cur_read;
    streams_cur_read.resize(batch_size);
    std::vector<std::string> results;
    results.resize(batch_size);

    torch::Tensor tail_padding =
	torch::zeros({static_cast<int32_t>(0.4 * sample_rate)}, torch::kFloat);

    std::vector<std::unique_ptr<sherpa::OnlineStream>> streams;
    for (int32_t i = 0; i != batch_size; ++i) {
	streams.push_back(online_asr.CreateStream());
    }

    std::vector<sherpa::OnlineStream *> ready_streams;
    std::vector<int32_t> ready_streams_id; // batch id for ready_stream
    while (true) {
	int32_t batch_samples_len = 0;
	// streaming input
	for (int32_t i = 0; i != batch_size; ++i) {
	    int32_t cur_frame_size = ((samples[i].size(0) - streams_cur_read[i]) < frame_size) ? (samples[i].size(0) - streams_cur_read[i]) : frame_size;
	    if (cur_frame_size > 0) {
		torch::Tensor cur_frame = samples[i].index({Slice(streams_cur_read[i], streams_cur_read[i] + cur_frame_size)});
		streams_cur_read[i] += cur_frame_size;
		batch_samples_len += cur_frame_size;
		streams[i]->AcceptWaveform(sample_rate, cur_frame);
		if (cur_frame_size < frame_size) {
		    streams[i]->AcceptWaveform(sample_rate, tail_padding);
		    streams[i]->InputFinished();
		    batch_samples_len += static_cast<int32_t>(0.4 * sample_rate);
		}
	    }
	}

	// batch decode
	while (true) {
	    ready_streams.clear();
	    ready_streams_id.clear();
	    for (int32_t i = 0; i != batch_size; ++i) {
		if (online_asr.IsReady(streams[i].get())) {
		    ready_streams.push_back(streams[i].get());
		    ready_streams_id.push_back(i);
		}
	    }
	    if (ready_streams.empty()) { break; }
	    online_asr.DecodeStreams(ready_streams.data(), ready_streams.size());

	    // update streaming decode results
	    for (int32_t j = 0; j != ready_streams.size(); ++j) {
		results[ready_streams_id[j]] += std::string("partial: ") + online_asr.GetResult(ready_streams[j]) + "\n";
		if (ready_streams[j]->IsEndpoint() || streams_cur_read[ready_streams_id[j]] == samples[ready_streams_id[j]].size(0)) {
		    results[ready_streams_id[j]] += std::string("final: ") + online_asr.GetResult(ready_streams[j]) + "\n";
		    // should reset the decoding instance when endpoint active
		    streams[ready_streams_id[j]] = online_asr.CreateStream();
		}
	    }
	}
	if (batch_samples_len == 0) { break; } 
    }
    return results;
}

int32_t main(int32_t argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::jit::getExecutorMode() = false;
  torch::jit::getProfilingMode() = false;
  torch::jit::setGraphExecutorOptimize(false);

  // All models in icefall use training data with sample rate 16000
  float expected_sample_rate = 16000;
  bool use_wav_scp = false;  // true to use wav.scp as input
  int32_t batch_size = 10;

  sherpa::ParseOptions po(kUsageMessage);

  sherpa::OnlineAsrOptions opts;
  opts.Register(&po);

  po.Register("use-wav-scp", &use_wav_scp,
              "If true, user should provide two arguments: "
              "scp:wav.scp ark,scp,t:results.ark,results.scp");

  po.Register("batch-size", &batch_size,
              "Used only when --use-wav-scp=true or --use-feats-scp=true. "
              "It specifies the batch size to use for decoding");

  po.Read(argc, argv);
  if (po.NumArgs() < 1) {
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  opts.Validate();

  SHERPA_CHECK_EQ(opts.fbank_opts.frame_opts.samp_freq, expected_sample_rate)
      << "The model was trained using training data with sample rate 16000. "
      << "We don't support resample yet";

  SHERPA_CHECK_GE(po.NumArgs(), 1);

  SHERPA_LOG(INFO) << "decoding method: " << opts.decoding_method;

  sherpa::OnlineAsr online_asr(opts);

  if (use_wav_scp) {
    SHERPA_CHECK_EQ(po.NumArgs(), 2)
        << "Please use something like:\n"
        << "scp:wav.scp ark,scp,t:results.scp,results.ark\n"
        << "if you provide --use-wav-scp=true";

    if (kaldiio::ClassifyRspecifier(po.GetArg(1), nullptr, nullptr) ==
        kaldiio::kNoRspecifier) {
      SHERPA_LOG(FATAL) << "Please provide an rspecifier. Current value is: "
                        << po.GetArg(1);
    }

    if (kaldiio::ClassifyWspecifier(po.GetArg(2), nullptr, nullptr, nullptr) ==
        kaldiio::kNoWspecifier) {
      SHERPA_LOG(FATAL) << "Please provide a wspecifier. Current value is: "
                        << po.GetArg(2);
    }

    SHERPA_CHECK_GT(batch_size, 0);

    kaldiio::TableWriter<kaldiio::TokenVectorHolder> writer(po.GetArg(2));

    kaldiio::SequentialTableReader<kaldiio::WaveHolder> wav_reader(
        po.GetArg(1));

    std::vector<std::string> keys;
    std::vector<torch::Tensor> values;
    for (; !wav_reader.Done(); wav_reader.Next()) {
      keys.push_back(wav_reader.Key());
      auto &wave_data = wav_reader.Value();
      if (wave_data.SampFreq() != expected_sample_rate) {
        SHERPA_LOG(FATAL) << wav_reader.Key()
                          << "is expected to have sample rate "
                          << expected_sample_rate << ". Given "
                          << wave_data.SampFreq();
      }
      auto &d = wave_data.Data();

      if (d.NumRows() > 1) {
        SHERPA_LOG(WARNING)
            << "Only the first channel from " << wav_reader.Key() << " is used";
      }

      auto tensor = torch::from_blob(const_cast<float *>(d.RowData(0)),
                                     {d.NumCols()}, torch::kFloat);
      values.push_back(tensor / 32768);

      if (keys.size() >= batch_size) {
        // now for recognition
        auto results = DecodeWaves(online_asr, values);
        for (size_t i = 0; i != keys.size(); ++i) {
          std::vector<std::string> words;
          kaldiio::SplitStringToVector(results[i], " ", true, &words);
          writer.Write(keys[i], words);
        }
        keys.clear();
        values.clear();
      }
    }  // for (; !wav_reader.Done(); wav_reader.Next())

    if (!keys.empty()) {
      auto results = DecodeWaves(online_asr, values);
      for (size_t i = 0; i != keys.size(); ++i) {
        std::vector<std::string> words;
        kaldiio::SplitStringToVector(results[i], " ", true, &words);
        writer.Write(keys[i], words);
      }
    }

    return 0;
  }  // if (use_wav_scp)

  std::vector<torch::Tensor> samples;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    std::string wave_filename = po.GetArg(i);
    torch::Tensor wave =
        sherpa::ReadWave(wave_filename, expected_sample_rate).first;
    samples.push_back(wave);
  }

  auto results = DecodeWaves(online_asr, samples);

  std::ostringstream os;
  for (int32_t i = 1; i <= po.NumArgs(); ++i) {
    os << "filename: " << po.GetArg(i) << "\n";
    os << "result: " << results[i - 1] << "\n\n";
  }
  std::cout << os.str();
}
