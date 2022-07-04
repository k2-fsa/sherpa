This is the folder to put torch.script.jit model and vocabulary dict.

* The torch.script.jit model could be exported from [icefall librispeech pruned_transducer_statelessX](https://github.com/k2-fsa/icefall/tree/master/egs/librispeech/ASR) recipes.
The exporting command is (taking pruned_transducer_stateless4 as an example):

```bash
epoch=29
avg=6
python pruned_transducer_stateless4/export.py \
    --exp-dir ./pruned_transducer_stateless4/exp \
    --epoch  ${epoch} \
    --avg ${avg} \
    --streaming-model 1 \
    --causal-convolution 1 \
    --jit 1
```

**Caution:** The torch.script.jit model **MUST** be named `jit.pt`.

**Note:** Only streaming transducer models in LibriSpeech are supported now (i.e. The model must be trained with `dynamic-chunk-training=1`), will add streaming model for Chinese datasets soon. 

* The vocabulary dict is a text file named `tokens.txt`. It has two columns as follows:

```
<blk> 0
<sos/eos> 1
<unk> 2
S 3
▁THE 4
▁A 5
T 6
▁AND 7
ED 8
▁OF 9
```

You can get this file from folder `egs/librispeech/ASR/data/lang_bpe_xxx` after running `prepare.sh`.

**Note:** You can only use models trained with bpe tokens now.


You can get pre-trained models from following links:

|  Model                                         |  path |
|------------------------------------------------|-------|
| pruned_transducer_stateless                    | https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless_20220625 |
| pruned_transducer_stateless2                   | https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless2_20220625 |
| pruned_transducer_stateless3 (giga_prob = 0.9) | https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless3_giga_0.9_20220625 |
| pruned_transducer_stateless4                   | https://huggingface.co/pkufool/icefall_librispeech_streaming_pruned_transducer_stateless4_20220625 |

The torch.script.jit model locates in `exp/cpu_jit-epoch-xx-avg-xx.pt` (**YOU HAVE TO RENAME IT TO** `jit.pt`), the vocabulary dict locates in `data/lang_bpe_500/tokens.txt`.
