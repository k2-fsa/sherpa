wav_dir=$1
wav_files=$(ls $wav_dir/*.wav)
# if wav_files is empty, then exit
if [ -z "$wav_files" ]; then
    exit 1
fi
hf_subset_name=$2
model_dir=./paraformer
model_path=$model_dir/sherpa-onnx-paraformer-zh-2023-09-14

if [ ! -d $model_path ]; then
    mkdir -p $model_dir
    wget -nc https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 -C $model_dir
fi

python3 scripts/offline-decode-files.py  \
    --tokens=$model_path/tokens.txt \
    --paraformer=$model_path/model.int8.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    --sample-rate=24000 \
    --log-dir $wav_dir \
    --feature-dim=80 \
    --label $hf_subset_name \
    $wav_files
