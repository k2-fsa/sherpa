# Introduction

This folder contains scripts for exporting models from
https://github.com/modelscope/3D-Speaker


Some of the exported models are listed below:

```
-rw-r--r--   1 runner  staff    29M Jan 10 02:50 3d_speaker-speech_campplus_sv_en_voxceleb_16k.pt
-rw-r--r--   1 runner  staff    28M Jan 10 02:50 3d_speaker-speech_campplus_sv_zh-cn_16k-common.pt
-rw-r--r--   1 runner  staff    28M Jan 10 02:50 3d_speaker-speech_campplus_sv_zh_en_16k-common_advanced.pt
-rw-r--r--   1 runner  staff    80M Jan 10 02:52 3d_speaker-speech_ecapa-tdnn_sv_en_voxceleb_16k.pt
-rw-r--r--   1 runner  staff    80M Jan 10 02:52 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k.pt
-rw-r--r--   1 runner  staff    80M Jan 10 02:51 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k.pt
-rw-r--r--   1 runner  staff    38M Jan 10 02:50 3d_speaker-speech_eres2net_base_200k_sv_zh-cn_16k-common.pt
-rw-r--r--   1 runner  staff    38M Jan 10 02:51 3d_speaker-speech_eres2net_base_sv_zh-cn_3dspeaker_16k.pt
-rw-r--r--   1 runner  staff   112M Jan 10 02:51 3d_speaker-speech_eres2net_large_sv_zh-cn_3dspeaker_16k.pt
-rw-r--r--   1 runner  staff    26M Jan 10 02:51 3d_speaker-speech_eres2net_sv_en_voxceleb_16k.pt
-rw-r--r--   1 runner  staff   212M Jan 10 02:50 3d_speaker-speech_eres2net_sv_zh-cn_16k-common.pt
-rw-r--r--   1 runner  staff    69M Jan 10 02:50 3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt
-rw-r--r--   1 runner  staff   206M Jan 10 02:50 3d_speaker-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.pt
```

```
./test.py --model 3d_speaker-speech_campplus_sv_en_voxceleb_16k.pt
----------testing 3d_speaker-speech_campplus_sv_en_voxceleb_16k.pt----------
embedding shape torch.Size([512])
tensor(0.6211) tensor(0.0356) tensor(0.0948)
----------testing 3d_speaker-speech_campplus_sv_en_voxceleb_16k.pt done----------


./test.py --model 3d_speaker-speech_campplus_sv_zh-cn_16k-common.pt
----------testing 3d_speaker-speech_campplus_sv_zh-cn_16k-common.pt----------
embedding shape torch.Size([192])
tensor(0.6936) tensor(-0.0842) tensor(0.0072)
----------testing 3d_speaker-speech_campplus_sv_zh-cn_16k-common.pt done----------

./test.py --model 3d_speaker-speech_campplus_sv_zh_en_16k-common_advanced.pt
----------testing 3d_speaker-speech_campplus_sv_zh_en_16k-common_advanced.pt----------
embedding shape torch.Size([192])
tensor(0.6668) tensor(0.0670) tensor(0.0569)
----------testing 3d_speaker-speech_campplus_sv_zh_en_16k-common_advanced.pt done----------

./test.py --model 3d_speaker-speech_ecapa-tdnn_sv_en_voxceleb_16k.pt
----------testing 3d_speaker-speech_ecapa-tdnn_sv_en_voxceleb_16k.pt----------
embedding shape torch.Size([192])
tensor(0.6733) tensor(-0.0007) tensor(0.0611)
----------testing 3d_speaker-speech_ecapa-tdnn_sv_en_voxceleb_16k.pt done----------

./test.py --model 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k.pt
----------testing 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k.pt----------
embedding shape torch.Size([192])
tensor(0.5880) tensor(0.1363) tensor(0.0885)
----------testing 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_3dspeaker_16k.pt done----------

./test.py --model 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k.pt
----------testing 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k.pt----------
embedding shape torch.Size([192])
tensor(0.7074) tensor(0.0289) tensor(0.1022)
----------testing 3d_speaker-speech_ecapa-tdnn_sv_zh-cn_cnceleb_16k.pt done----------

./test.py --model 3d_speaker-speech_eres2net_base_200k_sv_zh-cn_16k-common.pt
----------testing 3d_speaker-speech_eres2net_base_200k_sv_zh-cn_16k-common.pt----------
embedding shape torch.Size([512])
tensor(0.6675) tensor(0.0066) tensor(0.0576)
----------testing 3d_speaker-speech_eres2net_base_200k_sv_zh-cn_16k-common.pt done----------

./test.py --model 3d_speaker-speech_eres2net_base_sv_zh-cn_3dspeaker_16k.pt
----------testing 3d_speaker-speech_eres2net_base_sv_zh-cn_3dspeaker_16k.pt----------
embedding shape torch.Size([512])
tensor(0.6411) tensor(0.1044) tensor(0.0209)
----------testing 3d_speaker-speech_eres2net_base_sv_zh-cn_3dspeaker_16k.pt done----------

./test.py --model 3d_speaker-speech_eres2net_large_sv_zh-cn_3dspeaker_16k.pt
----------testing 3d_speaker-speech_eres2net_large_sv_zh-cn_3dspeaker_16k.pt----------
embedding shape torch.Size([512])
tensor(0.6336) tensor(0.0829) tensor(0.0681)
----------testing 3d_speaker-speech_eres2net_large_sv_zh-cn_3dspeaker_16k.pt done----------

./test.py --model 3d_speaker-speech_eres2net_sv_en_voxceleb_16k.pt
----------testing 3d_speaker-speech_eres2net_sv_en_voxceleb_16k.pt----------
embedding shape torch.Size([192])
tensor(0.6554) tensor(-0.0092) tensor(0.0551)
----------testing 3d_speaker-speech_eres2net_sv_en_voxceleb_16k.pt done----------

./test.py --model 3d_speaker-speech_eres2net_sv_zh-cn_16k-common.pt
----------testing 3d_speaker-speech_eres2net_sv_zh-cn_16k-common.pt----------
embedding shape torch.Size([192])
tensor(0.7127) tensor(0.0287) tensor(0.1308)
----------testing 3d_speaker-speech_eres2net_sv_zh-cn_16k-common.pt done----------

./test.py --model 3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt
----------testing 3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt----------
embedding shape torch.Size([192])
tensor(0.7194) tensor(0.0904) tensor(0.1441)
----------testing 3d_speaker-speech_eres2netv2_sv_zh-cn_16k-common.pt done----------

./test.py --model 3d_speaker-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.pt
----------testing 3d_speaker-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.pt----------
embedding shape torch.Size([192])
tensor(0.7625) tensor(-0.0190) tensor(0.1121)
----------testing 3d_speaker-speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common.pt done----------
```
