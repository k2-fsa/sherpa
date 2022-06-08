# File descriptions

## pruned_transducer_statelessX

Files in the part assume the model is from `pruned_transducer_statelessX` in
the folder <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR>
where `X>=2`.

| Filename | Description |
|----------|-------------|
| [conformer_rnnt/offline_server.py](./conformer_rnnt/offline_server.py) | The server for offline ASR |
| [conformer_rnnt/offline_client.py](./conformer/offline_client.py) | The client for offline ASR |
| [conformer_rnnt/decode_manifest.py](./conformer_rnnt/decode_manifest.py) | Demo for computing RTF and WER|

If you want to test the offline server without training your own model, you
can download pretrained models on the LibriSpeech corpus by visiting
<https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md>.
There you can find links to various pretrained models.

For instance, you can use <https://huggingface.co/csukuangfj/icefall-asr-librispeech-pruned-transducer-stateless3-2022-05-13>

## pruned_stateless_emformer_rnnt2

Files in the part assume the model is from `pruned_stateless_emformer_rnnt2` in
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

Files in the part assume the model is from `pruned_transducer_statelessX` in
the folder <https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR>
where `X>=2`. And the model is trained for streaming recognition.

| Filename | Description |
|----------|-------------|
| [streaming_conformer_rnnt/streaming_conformer_rnnt/streaming_server.py](./streaming_conformer_rnnt/streaming_server.py) | The server for streaming ASR |
| [streaming_conformer_rnnt/streaming_client.py](./streaming_conformer_rnnt/streaming_client.py) | The client for streaming ASR |
| [streaming_conformer_rnnt/decode.py](./streaming_conformer_rnnt/decode.py) | Utilities for streaming ASR|

TODO: Add pretrained streaming model.
