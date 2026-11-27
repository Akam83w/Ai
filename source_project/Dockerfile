FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 HF_HOME=/data/huggingface
COPY requirements-jais.txt /app/requirements-jais.txt
RUN pip install --no-cache-dir --prefer-binary -r /app/requirements-jais.txt
COPY model.py tokenizer.py api_server.py knowledge.py dataset.py train.py generate.py jais_api.py jais-chat.html /app/
COPY data/corpus /app/data/corpus
RUN mkdir -p /app/checkpoints /data/huggingface
EXPOSE 8080
CMD ["python", "api_server.py"]
