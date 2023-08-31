export CUDA_VISIBLE_DEVICES="0"

model_repo_path=./model_repo_whisper

encoder_fp32_file=./large-v2-encoder.onnx
decoder_fp32_file=./large-v2-decoder.onnx
encoder_file=./large-v2-encoder-fp16.onnx
decoder_file=./large-v2-decoder-fp16.onnx


python3 export-onnx-triton.py --model large-v2
polygraphy convert --fp-to-fp16 -o $encoder_file $encoder_fp32_file
polygraphy convert --fp-to-fp16 -o $decoder_file $decoder_fp32_file

cp $decoder_file $model_repo_path/decoder/1/
cp $encoder_file $model_repo_path/encoder/1/
