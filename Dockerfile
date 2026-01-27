FROM nvcr.io/nvidia/tritonserver:25.12-vllm-python-py3

ENV HF_HOME=/hf_cache

WORKDIR /workspace