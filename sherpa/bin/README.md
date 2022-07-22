# File descriptions

## pruned_transducer_statelessX

Files in this part assume the model is from `pruned_transducer_statelessX` in
the folder <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR>
where `X>=2`.

| Filename | Description |
|----------|-------------|
| [pruned_transducer_statelessX/offline_server.py](./pruned_transducer_statelessX/offline_server.py) | The server for offline ASR |
| [pruned_transducer_statelessX/offline_client.py](./pruned_transducer_statelessX/offline_client.py) | The client for offline ASR |
| [pruned_transducer_statelessX/decode_manifest.py](./pruned_transducer_statelessX/decode_manifest.py) | Demo for computing RTF and WER|

If you want to test the offline server without training your own model, you
can download pretrained models on the LibriSpeech corpus by visiting
<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>.
There you can find links to various pretrained models.

For instance, you can use <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>

## pruned_stateless_emformer_rnnt2

Files in this part assume the model is from `pruned_stateless_emformer_rnnt2` in
the folder <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR>.

| Filename | Description |
|----------|-------------|
| [pruned_stateless_emformer_rnnt2/streaming_server.py](./pruned_stateless_emformer_rnnt2/streaming_server.py) | The server for streaming ASR |
| [pruned_stateless_emformer_rnnt2/streaming_client.py](./pruned_stateless_emformer_rnnt2/streaming_client.py) | The client for streaming ASR |
| [pruned_stateless_emformer_rnnt2/decode.py](./pruned_stateless_emformer_rnnt2/decode.py) | Utilities for streaming ASR|

You can use the pretrained model from
<https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-stateless-emformer-rnnt2-2022-06-01>
to test it.

## Streaming pruned_transducer_statelessX

Files in this part assume the model is from `pruned_transducer_statelessX` in
the folder <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR>
where `X>=2`. And the model is trained for streaming recognition.

| Filename | Description |
|----------|-------------|
| [streaming_pruned_transducer_statelessX/streaming_server.py](./streaming_pruned_transducer_statelessX/streaming_server.py) | The server for streaming ASR |
| [streaming_pruned_transducer_statelessX/streaming_client.py](./streaming_pruned_transducer_statelessX/streaming_client.py) | The client for streaming ASR |
| [streaming_pruned_transducer_statelessX/decode.py](./streaming_pruned_transducer_statelessX/decode.py) | Utilities for streaming ASR|

You can use the pretrained model from
<https://huggingface.co/pkufool/icefall-asr-librispeech-pruned-stateless-streaming-conformer-rnnt4-2022-06-10>
to test it.
