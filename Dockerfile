from nvcr.io/nvidia/pytorch:22.12-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# Please choose previous pytorch:xx.xx if you encounter cuda driver mismatch issue
RUN pip3 install torchaudio
RUN pip3 install \
    k2==1.24.4.dev20240223+cuda11.7.torch1.13.1 -f https://k2-fsa.github.io/k2/cuda.html
# #install k2 from source
# #"sed -i ..."  line tries to turn off the cuda check
# RUN git clone https://github.com/k2-fsa/k2.git && \
#     cd k2 && \
#     sed -i 's/FATAL_ERROR/STATUS/g' cmake/torch.cmake && \
#     sed -i 's/in running_cuda_version//g' get_version.py && \
#     python3 setup.py install && \
#     cd -
WORKDIR /workspace

RUN git clone https://github.com/k2-fsa/icefall.git
ENV PYTHONPATH "${PYTHONPATH}:/workspace/icefall"
# https://github.com/k2-fsa/icefall/issues/674
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION "python"

RUN git clone https://github.com/k2-fsa/sherpa.git && \
    cd sherpa && \
    pip3 install -r ./requirements.txt && \
    python3 setup.py bdist_wheel && \
    pip3 install ./dist/k2_sherpa-*.whl --force


