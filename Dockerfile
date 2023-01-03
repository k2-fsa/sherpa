from nvcr.io/nvidia/pytorch:22.10-py3
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
# Please choose previous pytorch:xx.xx if you encounter cuda driver mismatch issue

#"sed -i ..."  line tries to turn off the cuda check
RUN git clone https://github.com/k2-fsa/k2.git && \
    cd k2 && \
    sed -i 's/FATAL_ERROR/STATUS/g' cmake/torch.cmake && \
    sed -i 's/in running_cuda_version//g' get_version.py && \
    python3 setup.py install && \
    cd -

RUN git clone https://github.com/k2-fsa/sherpa.git && \
    cd sherpa && \
    pip3 install -r ./requirements.txt && \
    python3 setup.py install --verbose


