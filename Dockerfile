FROM vllm/vllm-openai:v0.17.0

WORKDIR /workspace
COPY requirements.txt .

RUN pip install -r requirements.txt