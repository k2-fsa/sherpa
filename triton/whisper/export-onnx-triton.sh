export CUDA_VISIBLE_DEVICES="0"
name=large-v2
model_repo_path=./model_repo_whisper_${name}

encoder_fp32_file=./${name}-encoder.onnx
decoder_fp32_file=./${name}-decoder.onnx
encoder_file=./${name}-encoder-fp16.onnx
decoder_file=./${name}-decoder-fp16.onnx


python3 export-onnx-triton.py --model ${name}
polygraphy convert --fp-to-fp16 -o $encoder_file $encoder_fp32_file
polygraphy convert --fp-to-fp16 -o $decoder_file $decoder_fp32_file

cp $decoder_file $model_repo_path/decoder/1/
cp $encoder_file $model_repo_path/encoder/1/
